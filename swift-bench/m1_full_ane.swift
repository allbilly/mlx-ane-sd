// Standalone native Core ML runner for the public 0.6B M4-recipe adaptation.
// Build with swiftc -O -framework CoreML; no MLX/SwiftPM model downloads.
import Foundation
import CoreML

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }
func array(_ shape: [Int]) throws -> MLMultiArray {
    let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
    memset(a.dataPointer, 0, a.count * 2)
    return a
}
func pointer(_ a: MLMultiArray) -> UnsafeMutablePointer<UInt16> {
    precondition(a.dataType == .float16)
    return a.dataPointer.assumingMemoryBound(to: UInt16.self)
}
func copyVector(_ src: MLMultiArray, _ srcOffset: Int, _ srcStride: Int,
                _ dst: MLMultiArray, _ dstOffset: Int, _ dstStride: Int, _ n: Int) {
    let s = pointer(src).advanced(by: srcOffset), d = pointer(dst).advanced(by: dstOffset)
    if srcStride == 1 && dstStride == 1 { memcpy(d, s, n * 2) }
    else { for i in 0..<n { d[i * dstStride] = s[i * srcStride] } }
}
func fail(_ text: String) -> NSError { NSError(domain: "M1FullANE", code: 1, userInfo: [NSLocalizedDescriptionKey: text]) }

final class Program {
    let model: MLModel
    let role: String
    init(_ info: [String: Any], _ role: String) throws {
        let placement = info["placement"] as! [String: Any]
        guard placement["math_all_ane"] as? Bool == true else { throw fail("Unverified ANE placement: \(role)") }
        let config = MLModelConfiguration(); config.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: URL(fileURLWithPath: info["compiled"] as! String), configuration: config)
        self.role = role
    }
    func predict(_ inputs: [String: MLMultiArray], _ owner: Stack, _ label: String? = nil) throws -> [String: MLMultiArray] {
        // This command-line loop has no Cocoa run loop to drain temporary
        // feature providers/IOSurfaces. Keep returned arrays strongly retained
        // while releasing prediction temporaries after each call.
        return try autoreleasepool {
            let begin = now()
            let provider = try MLDictionaryFeatureProvider(dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
            let output = try model.prediction(from: provider)
            let key = label ?? role
            owner.profile[key, default: 0] += now() - begin
            owner.calls[key, default: 0] += 1
            var result = [String: MLMultiArray]()
            for name in output.featureNames {
                if let a = output.featureValue(for: name)?.multiArrayValue { result[name] = a }
            }
            return result
        }
    }
}

final class Chunk {
    let program: Program
    let count: Int, start: Int, captures: [Int]
    let k: MLMultiArray, v: MLMultiArray
    init(_ info: [String: Any], _ capacity: Int) throws {
        start = info["start"] as! Int; count = info["count"] as! Int; captures = info["captures"] as! [Int]
        program = try Program(info, "target\(start)")
        k = try array([count, 1, 8, capacity, 128]); v = try array([count, 1, 8, capacity, 128])
    }
}

final class Stack {
    let B = 8, W = 1024, H = 8, D = 128, F = 3072
    let C: Int, vocab: Int, draftLayers: Int, maskID: Int
    let featureIDs: [Int]
    let embedding: Data, embeddingOffset: Int
    let chunks: [Chunk], projector: Program, draft: Program, head: Program
    let dk: MLMultiArray, dv: MLMultiArray
    var offset = 0
    var profile = [String: Double](), calls = [String: Int]()
    var tables = [(MLMultiArray, MLMultiArray, MLMultiArray)]()

    init(_ directory: String) throws {
        let dir = URL(fileURLWithPath: directory)
        let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("manifest.json"))) as! [String: Any]
        guard manifest["status"] as? String == "complete", manifest["math_all_ane"] as? Bool == true else { throw fail("Incomplete artifacts") }
        let capacity = manifest["capacity"] as! Int
        C = capacity
        let config = manifest["config"] as! [String: Any]
        let dc = manifest["draft_config"] as! [String: Any]
        let transformer = dc["transformer_layer_config"] as! [String: Any]
        guard config["hidden_size"] as? Int == 1024, config["head_dim"] as? Int == 128,
              config["num_key_value_heads"] as? Int == 8, manifest["block"] as? Int == 8 else { throw fail("This runner requires the pinned 0.6B architecture") }
        vocab = config["vocab_size"] as! Int; draftLayers = transformer["num_hidden_layers"] as! Int
        maskID = dc["mask_token_id"] as! Int; featureIDs = dc["aux_hidden_state_layer_ids"] as! [Int]
        let embeddingData = try Data(contentsOf: dir.appendingPathComponent("embedding.npy"), options: .mappedIfSafe)
        embedding = embeddingData
        guard Array(embedding.prefix(6)) == [147, 78, 85, 77, 80, 89] else { throw fail("Invalid numpy embedding") }
        if embedding[6] == 1 { embeddingOffset = 10 + Int(embedding[8]) + (Int(embedding[9]) << 8) }
        else { embeddingOffset = 12 + (0..<4).reduce(0) { $0 + (Int(embeddingData[8 + $1]) << (8 * $1)) } }
        guard embedding.count == embeddingOffset + vocab * W * 2 else { throw fail("Unexpected embedding layout") }
        let infos = manifest["artifacts"] as! [[String: Any]]
        chunks = try infos.filter { $0["role"] as? String == "target" }.map { try Chunk($0, capacity) }
        projector = try Program(infos.first { $0["role"] as? String == "projector" }!, "projector")
        draft = try Program(infos.first { $0["role"] as? String == "draft" }!, "draft")
        head = try Program(infos.first { $0["role"] as? String == "head" }!, "head")
        dk = try array([draftLayers, 1, H, C, D]); dv = try array([draftLayers, 1, H, C, D])
        let theta = config["rope_theta"] as! Double
        for pos in 0...C {
            let cos = try array([1, 1, B, D]), sin = try array([1, 1, B, D]), mask = try array([1, 1, B, C + B])
            for row in 0..<B {
                for col in 0..<(D / 2) {
                    let angle = Double(pos + row) * pow(theta, -Double(col) / Double(D / 2))
                    let cv = Float16(Foundation.cos(angle)).bitPattern, sv = Float16(Foundation.sin(angle)).bitPattern
                    for column in [col, col + D / 2] { pointer(cos)[row * D + column] = cv; pointer(sin)[row * D + column] = sv }
                }
                for col in 0..<(C + B) {
                    pointer(mask)[row * (C + B) + col] = Float16(col < pos || (col >= C && col - C <= row) ? 0 : -10000).bitPattern
                }
            }
            tables.append((cos, sin, mask))
        }
    }

    func reset() {
        offset = 0; profile = [:]; calls = [:]
        for c in chunks { memset(c.k.dataPointer, 0, c.k.count * 2); memset(c.v.dataPointer, 0, c.v.count * 2) }
        memset(dk.dataPointer, 0, dk.count * 2); memset(dv.dataPointer, 0, dv.count * 2)
    }

    func embed(_ ids: [Int]) throws -> MLMultiArray {
        let hidden = try array([1, B, W])
        embedding.withUnsafeBytes { bytes in
            for (row, id) in ids.enumerated() {
                precondition(id >= 0 && id < vocab)
                memcpy(pointer(hidden).advanced(by: row * W), bytes.baseAddress!.advanced(by: embeddingOffset + id * W * 2), W * 2)
            }
        }
        return hidden
    }

    func ids(_ hidden: MLMultiArray, _ rows: Range<Int>, _ label: String) throws -> [Int] {
        let outputs = try head.predict(["hidden": hidden], self, label)
        let begin = now()
        var ids = [Int](repeating: 0, count: rows.count), best = [Float](repeating: -.infinity, count: rows.count)
        for tile in 0..<((vocab + 8191) / 8192) {
            let a = outputs["logits\(tile)"]!, stride = a.strides.map { $0.intValue }, cols = a.shape[2].intValue
            let p = pointer(a)
            for (local, row) in rows.enumerated() {
                let base = row * stride[1]
                for col in 0..<cols {
                    let value = Float(Float16(bitPattern: p[base + col * stride[2]]))
                    if value > best[local] { best[local] = value; ids[local] = tile * 8192 + col }
                }
            }
        }
        guard best.allSatisfy({ $0.isFinite }) else { throw fail("Non-finite vocabulary logits") }
        profile["host_argmax", default: 0] += now() - begin
        return ids
    }

    struct Forward {
        let ids: [Int], features: MLMultiArray, pending: [[String: MLMultiArray]]
    }
    func forward(_ tokenIDs: [Int], _ logits: Bool = true) throws -> Forward {
        guard !tokenIDs.isEmpty && tokenIDs.count <= B && offset + tokenIDs.count <= C else { throw fail("Context/block limit") }
        var hidden = try embed(tokenIDs), pending = [[String: MLMultiArray]]()
        let features = try array([1, B, F]), (cos, sin, mask) = tables[offset]
        for c in chunks {
            let out = try c.program.predict(["hidden": hidden, "cos": cos, "sin": sin, "mask": mask, "cache_k": c.k, "cache_v": c.v], self)
            hidden = out["hidden_out"]!; pending.append(out)
            if !c.captures.isEmpty {
                let capture = out["captures"]!, s = capture.strides.map { $0.intValue }
                for (local, index) in c.captures.enumerated() {
                    let feature = featureIDs.firstIndex(of: index)!
                    for row in 0..<tokenIDs.count { copyVector(capture, local * s[0] + row * s[2], s[3], features, row * F + feature * W, 1, W) }
                }
            }
        }
        return Forward(ids: logits ? try ids(hidden, 0..<tokenIDs.count, "target_head") : [], features: features, pending: pending)
    }

    func copyKV(_ new: MLMultiArray, _ cache: MLMultiArray, _ count: Int, _ pos: Int) {
        let s = new.strides.map { $0.intValue }, d = cache.strides.map { $0.intValue }
        for layer in 0..<cache.shape[0].intValue {
            for head in 0..<H { for row in 0..<count {
                copyVector(new, layer * s[0] + head * s[2] + row * s[3], s[4],
                           cache, layer * d[0] + head * d[2] + (pos + row) * d[3], d[4], D)
            } }
        }
    }
    func commit(_ out: Forward, _ count: Int) {
        for (c, pending) in zip(chunks, out.pending) {
            copyKV(pending["new_k"]!, c.k, count, offset); copyKV(pending["new_v"]!, c.v, count, offset)
        }
        offset += count
    }
    func append(_ features: MLMultiArray, _ count: Int, _ pos: Int) throws {
        let input = try array([1, B, F])
        for row in 0..<count { copyVector(features, row * F, 1, input, row * F, 1, F) }
        let (cos, sin, _) = tables[pos]
        let out = try projector.predict(["features": input, "cos": cos, "sin": sin], self)
        copyKV(out["new_k"]!, dk, count, pos); copyKV(out["new_v"]!, dv, count, pos)
    }
    func propose(_ anchor: Int) throws -> [Int] {
        let (cos, sin, mask) = tables[offset]
        let out = try draft.predict(["hidden": embed([anchor] + [Int](repeating: maskID, count: B - 1)), "cos": cos, "sin": sin,
                                     "cache_k": dk, "cache_v": dv, "mask": mask], self)
        return try ids(out["hidden_out"]!, 1..<B, "draft_head")
    }

    func generate(_ prompt: [Int], _ limit: Int, _ speculative: Bool, _ eos: Set<Int>) throws -> [String: Any] {
        guard !prompt.isEmpty && prompt.count + limit <= C && limit > 0 else { throw fail("Prompt/generation limit") }
        reset(); let prefillStart = now()
        for start in stride(from: 0, to: prompt.count - 1, by: B) {
            let part = Array(prompt[start..<min(start + B, prompt.count - 1)]), pos = offset
            let out = try forward(part, false); commit(out, part.count)
            if speculative { try append(out.features, part.count, pos) }
        }
        let prefill = now() - prefillStart
        profile = [:]; calls = [:]; let begin = now(), pos = offset
        let first = try forward([prompt.last!]); commit(first, 1)
        if speculative { try append(first.features, 1, pos) }
        var tokens = [first.ids[0]], traces = [[String: Any]](), accepted = 0, proposed = 0
        while tokens.count < limit && !eos.contains(tokens.last!) {
            let candidates = speculative && limit - tokens.count > 1 ? Array(try propose(tokens.last!).prefix(limit - tokens.count - 1)) : []
            let pos = offset, out = try forward([tokens.last!] + candidates)
            var count = 0
            for (candidate, prediction) in zip(candidates, out.ids) { if candidate != prediction { break }; count += 1 }
            var committed = Array(candidates.prefix(count)) + [out.ids[count]]
            if let stop = committed.firstIndex(where: { eos.contains($0) }) { committed = Array(committed.prefix(stop + 1)) }
            commit(out, committed.count)
            if speculative { try append(out.features, committed.count, pos) }
            tokens += committed; accepted += min(count, committed.count); proposed += candidates.count
            traces.append(["candidates": candidates, "predictions": out.ids, "accepted": count, "committed": committed, "offset": offset])
        }
        let elapsed = now() - begin
        return ["tokens": tokens, "decode_s": elapsed, "prefill_s": prefill, "tok_per_s_decode": Double(tokens.count) / elapsed,
                "cycles": traces.count, "accepted": accepted, "proposed": proposed, "profile_s": profile, "calls": calls, "trace": traces]
    }
}

func emit(_ object: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    print(String(data: data, encoding: .utf8)!); fflush(stdout)
}
do {
    let begin = now(), stack = try Stack(CommandLine.arguments[1])
    try emit(["ready": true, "load_s": now() - begin])
    while let line = readLine() {
        do {
            try autoreleasepool {
                let request = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
                let result = try stack.generate(request["prompt_ids"] as! [Int], request["limit"] as! Int,
                                                request["speculative"] as! Bool, Set(request["eos"] as! [Int]))
                try emit(result)
            }
        } catch { try emit(["error": String(describing: error)]) }
    }
} catch { fputs("\(error)\n", stderr); exit(1) }
