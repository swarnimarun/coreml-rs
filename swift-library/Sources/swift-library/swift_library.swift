import CoreML

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
			let res = self.description!.outputDescriptionsByName[name.toString()]!
			let ret = RustVec<UInt>()
			for r in res.multiArrayConstraint!.shape {
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
	init(output: [String: Any]?, error: (any Error)?) {
		self.output = output
		self.error = error
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
		let out = (output[name.toString()]! as? MLMultiArray)!
		let l = out.count
		var v = RustVec<Float32>()
		out.withUnsafeMutableBytes { ptr, strides in
			let p = ptr.baseAddress!.assumingMemoryBound(to: Float32.self)
			v = rust_vec_from_ptr_f32(p, UInt(l))
		}
		return v
	}
	func outputI32(name: RustString) -> RustVec<Int32> {
		if hasFailedToLoad() { return RustVec.init() }
		let output = self.output!
		let out = (output[name.toString()]! as? MLMultiArray)!
		let l = out.count
		var v = RustVec<Int32>()
		out.withUnsafeMutableBytes { ptr, strides in
			let p = ptr.baseAddress!.assumingMemoryBound(to: Int32.self)
			v = rust_vec_from_ptr_i32(p, UInt(l))
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
			v = rust_vec_from_ptr_u16(p, UInt(l))
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
		let m = Model.init(failedToLoad: false)
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
		if hasFailedToLoad() { return ModelOutput(output: nil, error: RuntimeError("Failed to load model; can't run predict")) }
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
		shape: RustVec<Int32>, featureName: RustString, data: UnsafeMutablePointer<Float32>,
		len: UInt
	) -> Bool {
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
		shape: RustVec<Int32>, featureName: RustString, data: UnsafeMutablePointer<Int32>, len: UInt
	) -> Bool {
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
		shape: RustVec<Int32>, featureName: RustString, data: UnsafeMutablePointer<UInt16>,
		len: UInt
	) -> Bool {
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
