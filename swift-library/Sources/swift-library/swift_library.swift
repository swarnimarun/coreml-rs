import CoreML

class BatchOutput {
	var batchProvider: MLBatchProvider? = nil
	var error: String? = nil
	init(error: String? = nil, batchProvider: MLBatchProvider? = nil) {
		self.batchProvider = batchProvider
		self.error = error
	}

	func getOutputAtIndex(at: Int) -> ModelOutput {
		let features = self.batchProvider?.features(at: at) as? MLDictionaryFeatureProvider
		return ModelOutput.init(output: features?.dictionary, cpy: true)
	}

	func count() -> Int {
		let c = self.batchProvider?.count
		guard let c else { return 0 }
		return c
	}

	func getError() -> RustString? {
		if self.error == nil {
			return nil
		}
		return "\(self.error!)".intoRustString()
	}
}

class BatchModelInput {
	var dict: [String: Any] = [:]
	func toFeatureProvider() -> MLDictionaryFeatureProvider? {
		do {
			return try MLDictionaryFeatureProvider.init(dictionary: self.dict)
		} catch {
			return nil
		}
	}
}

class BatchModel: @unchecked Sendable {
	var compiledPath: URL? = nil
	var model: MLModel? = nil
	var modelCompiledAsset: MLModelAsset? = nil
	var inputs: [BatchModelInput] = []
	var computeUnits: MLComputeUnits = .cpuAndNeuralEngine
	var failedToLoad: Bool

	init(failedToLoad: Bool = false, model: MLModel? = nil) {
		self.failedToLoad = failedToLoad
	}

	func hasFailedToLoad() -> Bool {
		return self.failedToLoad
	}

	func description() -> ModelDescription {
		return ModelDescription(desc: self.model?.modelDescription)
	}

	func load() -> Bool {
		if hasFailedToLoad() { return false }
		let config = MLModelConfiguration.init()
		config.computeUnits = self.computeUnits
		do {
			if self.compiledPath == nil {
				let semaphore = DispatchSemaphore(value: 0)
				Task { [weak self] in
					guard let self else { return }
					let asset = self.modelCompiledAsset!
					let res = try await MLModel.load(asset: asset, configuration: config)
					self.model = res
					semaphore.signal()
				}
				semaphore.wait()
			} else {
				let loadedModel = try MLModel(contentsOf: self.compiledPath!, configuration: config)
				self.model = loadedModel
			}
			return true
		} catch {
			return false
		}
	}

	func unload() -> Bool {
		if hasFailedToLoad() { return false }
		self.model = nil
		return true
	}

	func bindInputF32(
		shape: RustVec<UInt>, featureName: RustString, data: UnsafeMutablePointer<Float32>,
		len: UInt, idx: Int
	) -> Bool {
		do {
			var arr: [NSNumber] = []
			var stride: [NSNumber] = []
			var m: UInt = 1
			for i in shape.reversed() {
				stride.append(NSNumber(value: m))
				m = i * m
			}
			stride.reverse()
			for s in shape {
				arr.append(NSNumber(value: s))
			}
			let deallocMultiArrayRust = { (_ ptr: UnsafeMutableRawPointer) in
				rust_vec_free_f32(ptr.assumingMemoryBound(to: Float32.self), len)
			}
			let array = try MLMultiArray.init(
				dataPointer: data, shape: arr, dataType: MLMultiArrayDataType.float32,
				strides: stride, deallocator: deallocMultiArrayRust)
			let value = MLFeatureValue(multiArray: array)
			if self.inputs.count <= idx {
				self.inputs.append(BatchModelInput.init())
			}
			self.inputs[idx].dict[featureName.toString()] = value
			return true
		} catch {
			print("Unexpected input error; \(error)")
			return false
		}
	}

	func predict() -> BatchOutput {
		do {
			let opts = MLPredictionOptions.init()
			// TODO (SA): to feature provider
			let features = inputs.compactMap { input in
				input.toFeatureProvider()
			}
			let batchProvider = MLArrayBatchProvider.init(array: features)
			let output = try self.model?.predictions(from: batchProvider, options: opts)
			guard let output else {
				return BatchOutput.init(error: "ran predict without a model loaded into memory")
			}
			return BatchOutput.init(batchProvider: output)
		} catch {
			return BatchOutput.init(error: error.localizedDescription)
		}
	}
}

class ModelDescription {
	var description: MLModelDescription? = nil
	init(desc: MLModelDescription?) {
		self.description = desc
	}

	func failedToLoad() -> Bool { return self.description == nil }

	func inputs() -> RustVec<RustString> {
		let ret = RustVec<RustString>()
		if !failedToLoad() {
			for (_, value) in self.description!.inputDescriptionsByName {
				let str = "\(value)".intoRustString()
				ret.push(value: str)
			}
		}
		return ret
	}
	func outputs() -> RustVec<RustString> {
		let ret = RustVec<RustString>()
		if !failedToLoad() {
			for (_, value) in self.description!.outputDescriptionsByName {
				let str = "\(value)".intoRustString()
				ret.push(value: str)
			}
		}
		return ret
	}
	func output_type(name: RustString) -> RustString {
		if !failedToLoad() {
			let res = self.description!.outputDescriptionsByName[name.toString()]!
			if res.multiArrayConstraint!.dataType == MLMultiArrayDataType.float32 {
				return "f32".intoRustString()
			}
		}
		return "".intoRustString()
	}
	func output_shape(name: RustString) -> RustVec<UInt> {
		if !failedToLoad() {
			let res = self.description?.outputDescriptionsByName[name.toString()]
			guard let res else { return RustVec.init() }
			let arr = res.multiArrayConstraint
			guard let arr else { return RustVec.init() }
			let ret = RustVec<UInt>()
			for r in arr.shape {
				ret.push(value: UInt(truncating: r))
			}
			return ret
		}
		return RustVec.init()
	}
	func input_shape(name: RustString) -> RustVec<UInt> {
		if !failedToLoad() {
			let res = self.description?.inputDescriptionsByName[name.toString()]
			guard let res else { return RustVec.init() }
			let arr = res.multiArrayConstraint
			guard let arr else { return RustVec.init() }
			let ret = RustVec<UInt>()
			for r in arr.shape {
				ret.push(value: UInt(truncating: r))
			}
			return ret
		}
		return RustVec.init()
	}

	func output_names() -> RustVec<RustString> {
		if !failedToLoad() {
			let ret = RustVec<RustString>()
			for (key, _) in self.description!.outputDescriptionsByName {
				ret.push(value: key.intoRustString())
			}
			return ret
		}
		return RustVec.init()
	}
}

class ModelOutput {
	var output: [String: Any]? = [:]
	var error: (any Error)? = nil
	var cpy: Bool = false
	init(output: [String: Any]?, error: (any Error)? = nil, cpy: Bool = false) {
		self.output = output
		self.error = error
		self.cpy = cpy
	}
	func hasFailedToLoad() -> Bool {
		return self.error != nil
	}
	func getError() -> RustString? {
		if self.error == nil {
			return nil
		}
		return "\(self.error!)".intoRustString()
	}
	func outputDescription() -> RustVec<RustString> {
		if hasFailedToLoad() { return RustVec.init() }
		let output = self.output!
		let ret = RustVec<RustString>()
		for key in output.keys {
			let str = "\(key):\(output[key]!)".intoRustString()
			ret.push(value: str)
		}
		return ret
	}
	func outputF32(name: RustString) -> RustVec<Float32> {
		if hasFailedToLoad() { return RustVec.init() }
		let output = self.output!
		if self.cpy {
			let out = (output[name.toString()]! as? MLFeatureValue)!.multiArrayValue!
			let l = out.count
			var v = RustVec<Float32>()
			print("outputF32: ", name.toString())
			out.withUnsafeMutableBytes { ptr, strides in
				let p = ptr.baseAddress!.assumingMemoryBound(to: Float32.self)
				v = rust_vec_from_ptr_f32_cpy(p, UInt(l))
			}
			return v
		} else {
			let out = (output[name.toString()]! as? MLMultiArray)!
			let l = out.count
			var v = RustVec<Float32>()
			out.withUnsafeMutableBytes { ptr, strides in
				let p = ptr.baseAddress!.assumingMemoryBound(to: Float32.self)
				v = rust_vec_from_ptr_f32(p, UInt(l))
			}
			return v
		}

	}
	func outputI32(name: RustString) -> RustVec<Int32> {
		if hasFailedToLoad() { return RustVec.init() }
		let output = self.output!
		let out = (output[name.toString()]! as? MLMultiArray)!
		let l = out.count
		var v = RustVec<Int32>()
		out.withUnsafeMutableBytes { ptr, strides in
			let p = ptr.baseAddress!.assumingMemoryBound(to: Int32.self)
			if self.cpy {
				v = rust_vec_from_ptr_i32_cpy(p, UInt(l))
			} else {
				v = rust_vec_from_ptr_i32(p, UInt(l))
			}
		}
		return v
	}
	func outputU16(name: RustString) -> RustVec<UInt16> {
		if hasFailedToLoad() { return RustVec.init() }
		let output = self.output!
		let out = (output[name.toString()]! as? MLMultiArray)!
		let l = out.count
		var v = RustVec<UInt16>()
		out.withUnsafeMutableBytes { ptr, strides in
			let p = ptr.baseAddress!.assumingMemoryBound(to: UInt16.self)
			if self.cpy {
				v = rust_vec_from_ptr_u16_cpy(p, UInt(l))
			} else {
				v = rust_vec_from_ptr_u16(p, UInt(l))
			}
		}
		return v
	}
}

func initWithCompiledAsset(
	ptr: UnsafeMutablePointer<UInt8>, len: Int, compute: ComputePlatform
) -> Model {
	var computeUnits: MLComputeUnits
	switch compute {
	case .Cpu:
		computeUnits = .cpuOnly
		break
	case .CpuAndANE:
		computeUnits = .cpuAndNeuralEngine
		break
	case .CpuAndGpu:
		computeUnits = .cpuAndGPU
		break
	}
	let data = Data.init(
		bytesNoCopy: ptr, count: len,
		deallocator: Data.Deallocator.custom { ptr, len in
			rust_vec_free_u8(ptr.assumingMemoryBound(to: UInt8.self), UInt(len))
		})
	do {
		let m = Model.init(failedToLoad: false)
		m.modelCompiledAsset = try MLModelAsset.init(specification: data)
		m.computeUnits = computeUnits
		return m
	} catch {
		let m = Model.init(failedToLoad: true)
		return m
	}
}

func initWithCompiledAssetBatch(
	ptr: UnsafeMutablePointer<UInt8>, len: Int, compute: ComputePlatform
) -> BatchModel {
	var computeUnits: MLComputeUnits
	switch compute {
	case .Cpu:
		computeUnits = .cpuOnly
		break
	case .CpuAndANE:
		computeUnits = .cpuAndNeuralEngine
		break
	case .CpuAndGpu:
		computeUnits = .cpuAndGPU
		break
	}
	let data = Data.init(
		bytesNoCopy: ptr, count: len,
		deallocator: Data.Deallocator.custom { ptr, len in
			rust_vec_free_u8(ptr.assumingMemoryBound(to: UInt8.self), UInt(len))
		})
	do {
		let m = BatchModel.init(failedToLoad: false)
		m.modelCompiledAsset = try MLModelAsset.init(specification: data)
		m.computeUnits = computeUnits
		return m
	} catch {
		let m = BatchModel.init(failedToLoad: true)
		return m
	}
}

func initWithPath(path: RustString, compute: ComputePlatform, compiled: Bool) -> Model {
	var computeUnits: MLComputeUnits
	switch compute {
	case .Cpu:
		computeUnits = .cpuOnly
		break
	case .CpuAndANE:
		computeUnits = .cpuAndNeuralEngine
		break
	case .CpuAndGpu:
		computeUnits = .cpuAndGPU
		break
	}
	var compiledPath: URL
	if compiled {
		compiledPath = URL(string: path.toString())!
	} else {
		let url = URL(string: path.toString())!
		do {
			compiledPath = try MLModel.compileModel(at: url)
		} catch {
			return Model.init(failedToLoad: true)
		}
	}
	let m = Model.init(failedToLoad: false)
	m.compiledPath = compiledPath
	m.computeUnits = computeUnits
	return m
}

func initWithPathBatch(path: RustString, compute: ComputePlatform, compiled: Bool) -> BatchModel {
	var computeUnits: MLComputeUnits
	switch compute {
	case .Cpu:
		computeUnits = .cpuOnly
		break
	case .CpuAndANE:
		computeUnits = .cpuAndNeuralEngine
		break
	case .CpuAndGpu:
		computeUnits = .cpuAndGPU
		break
	}
	var compiledPath: URL
	if compiled {
		compiledPath = URL(string: path.toString())!
	} else {
		let url = URL(string: path.toString())!
		do {
			compiledPath = try MLModel.compileModel(at: url)
		} catch {
			return BatchModel.init(failedToLoad: true)
		}
	}
	let m = BatchModel.init(failedToLoad: false)
	m.compiledPath = compiledPath
	m.computeUnits = computeUnits
	return m
}

struct RuntimeError: LocalizedError {
	let description: String

	init(_ description: String) {
		self.description = description
	}

	var errorDescription: String? {
		description
	}
}

class Model: @unchecked Sendable {
	var compiledPath: URL? = nil
	var modelCompiledAsset: MLModelAsset? = nil
	var model: MLModel? = nil
	var dict: [String: Any] = [:]
	var outputs: [String: Any] = [:]
	var computeUnits: MLComputeUnits = .cpuAndNeuralEngine

	var failedToLoad: Bool
	init(failedToLoad: Bool) {
		self.failedToLoad = failedToLoad
	}

	func getCompiledPath() -> RustString? {
		return self.compiledPath?.absoluteString.intoRustString()
	}

	func hasFailedToLoad() -> Bool {
		return self.failedToLoad
	}

	func load() -> Bool {
		if hasFailedToLoad() { return false }
		let config = MLModelConfiguration.init()
		config.computeUnits = self.computeUnits
		do {
			if self.compiledPath == nil {
				let semaphore = DispatchSemaphore(value: 0)
				Task { [weak self] in
					guard let self else { return }
					let asset = self.modelCompiledAsset!
					let res = try await MLModel.load(asset: asset, configuration: config)
					self.model = res
					semaphore.signal()
				}
				semaphore.wait()
			} else {
				let loadedModel = try MLModel(contentsOf: self.compiledPath!, configuration: config)
				self.model = loadedModel
			}
			return true
		} catch {
			return false
		}
	}

	func unload() -> Bool {
		if hasFailedToLoad() { return false }
		self.model = nil
		return true
	}

	func description() -> ModelDescription {
		return ModelDescription(desc: self.model?.modelDescription)
	}

	func bindOutputF32(
		shape: RustVec<Int32>, featureName: RustString, data: UnsafeMutablePointer<Float32>,
		len: UInt
	) -> Bool {
		if hasFailedToLoad() { return false }
		do {
			var arr: [NSNumber] = []
			var stride: [NSNumber] = []
			var m: Int32 = 1
			for i in shape.reversed() {
				stride.append(NSNumber(value: m))
				m = i * m
			}
			stride.reverse()
			for s in shape {
				arr.append(NSNumber(value: s))
			}
			let deallocMultiArrayRust = { (_ ptr: UnsafeMutableRawPointer) in
				()
			}
			let array = try MLMultiArray.init(
				dataPointer: data, shape: arr, dataType: MLMultiArrayDataType.float32,
				strides: stride, deallocator: deallocMultiArrayRust)
			self.outputs[featureName.toString()] = array
			return true
		} catch {
			print("Unexpected output error: \(error)")
			return false
		}
	}

	func predict() -> ModelOutput {
		if hasFailedToLoad() {
			return ModelOutput(
				output: nil, error: RuntimeError("Failed to load model; can't run predict"))
		}
		do {
			let input = try MLDictionaryFeatureProvider.init(dictionary: self.dict)
			let opts = MLPredictionOptions.init()
			opts.outputBackings = self.outputs
			try self.model!.prediction(from: input, options: opts)
			let outputs = self.outputs
			self.outputs = [:]
			self.dict = [:]
			return ModelOutput(output: outputs, error: nil)
		} catch {
			// print("Unexpected predict error: \(error)")
			return ModelOutput(output: nil, error: error)
		}
	}

	func bindInputF32(
		shape: RustVec<UInt>, featureName: RustString, data: UnsafeMutablePointer<Float32>,
		len: UInt
	) -> Bool {
		do {
			var arr: [NSNumber] = []
			var stride: [NSNumber] = []
			var m: UInt = 1
			for i in shape.reversed() {
				stride.append(NSNumber(value: m))
				m = i * m
			}
			stride.reverse()
			for s in shape {
				arr.append(NSNumber(value: s))
			}
			let deallocMultiArrayRust = { (_ ptr: UnsafeMutableRawPointer) in
				rust_vec_free_f32(ptr.assumingMemoryBound(to: Float32.self), len)
			}
			let array = try MLMultiArray.init(
				dataPointer: data, shape: arr, dataType: MLMultiArrayDataType.float32,
				strides: stride, deallocator: deallocMultiArrayRust)
			let value = MLFeatureValue(multiArray: array)
			self.dict[featureName.toString()] = value
			return true
		} catch {
			print("Unexpected input error; \(error)")
			return false
		}
	}

	func bindInputI32(
		shape: RustVec<UInt>, featureName: RustString, data: UnsafeMutablePointer<Int32>, len: UInt
	) -> Bool {
		do {
			var arr: [NSNumber] = []
			var stride: [NSNumber] = []
			var m: UInt = 1
			for i in shape.reversed() {
				stride.append(NSNumber(value: m))
				m = i * m
			}
			stride.reverse()
			for s in shape {
				arr.append(NSNumber(value: s))
			}
			let deallocMultiArrayRust = { (_ ptr: UnsafeMutableRawPointer) -> Void in
				rust_vec_free_i32(ptr.assumingMemoryBound(to: Int32.self), len)
			}
			let array = try MLMultiArray.init(
				dataPointer: data, shape: arr, dataType: MLMultiArrayDataType.float32,
				strides: stride, deallocator: deallocMultiArrayRust)
			let value = MLFeatureValue(multiArray: array)
			self.dict[featureName.toString()] = value
			return true
		} catch {
			print("Unexpected error; \(error)")
			return false
		}
	}

	func bindInputU16(
		shape: RustVec<UInt>, featureName: RustString, data: UnsafeMutablePointer<UInt16>,
		len: UInt
	) -> Bool {
		do {
			var arr: [NSNumber] = []
			var stride: [NSNumber] = []
			var m: UInt = 1
			for i in shape.reversed() {
				stride.append(NSNumber(value: m))
				m = i * m
			}
			stride.reverse()
			for s in shape {
				arr.append(NSNumber(value: s))
			}
			let deallocMultiArrayRust = { (_ ptr: UnsafeMutableRawPointer) -> Void in
				rust_vec_free_u16(ptr.assumingMemoryBound(to: UInt16.self), len)
			}
			let array = try MLMultiArray.init(
				dataPointer: data, shape: arr, dataType: MLMultiArrayDataType.float16,
				strides: stride, deallocator: deallocMultiArrayRust)
			let value = MLFeatureValue(multiArray: array)
			self.dict[featureName.toString()] = value
			return true
		} catch {
			print("Unexpected error; \(error)")
			return false
		}
	}
}
