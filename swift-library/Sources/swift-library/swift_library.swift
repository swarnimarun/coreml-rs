import CoreML

// Load and Save Data
func LoadAndSaveData(path: RustString, to: RustString) {
	let data = try! Data.init(contentsOf: URL(string: path.toString())!)
	try! data.write(to: URL(string: to.toString())!)
}

class ModelDescription {
	var description: MLModelDescription? = nil
	init(desc: MLModelDescription) {
		self.description = desc
	}
	func inputs() -> RustVec<RustString> {
		let ret = RustVec<RustString>()
		for (_, value) in self.description!.inputDescriptionsByName {
			let str = "\(value)".intoRustString()
			ret.push(value: str)
		}
		return ret
	}
	func outputs() -> RustVec<RustString> {
		let ret = RustVec<RustString>()
		for (_, value) in self.description!.outputDescriptionsByName {
			let str = "\(value)".intoRustString()
			ret.push(value: str)
		}
		return ret
	}
	func output_type(name: RustString) -> RustString {
		let res = self.description!.outputDescriptionsByName[name.toString()]!
		if res.multiArrayConstraint!.dataType == MLMultiArrayDataType.float32 {
			return "f32".intoRustString()
		}
		return "".intoRustString()
	}
	func output_shape(name: RustString) -> RustVec<UInt> {
		let res = self.description!.outputDescriptionsByName[name.toString()]!
		let ret = RustVec<UInt>()
		for r in res.multiArrayConstraint!.shape {
			ret.push(value: UInt(truncating: r))
		}
		return ret
	}
	func output_names() -> RustVec<RustString> {
		let ret = RustVec<RustString>()
		for (key, _) in self.description!.outputDescriptionsByName {
			ret.push(value: key.intoRustString())
		}
		return ret
	}
}

class ModelOutput {
	var output: [String: Any] = [:]
	init(output: [String: Any]) {
		self.output = output
	}
	func outputDescription() -> RustVec<RustString> {
		let ret = RustVec<RustString>()
		for key in self.output.keys {
			let str = "\(key):\(self.output[key]!)".intoRustString()
			ret.push(value: str)
		}
		return ret
	}
	func outputF32(name: RustString) -> RustVec<Float32> {
		let output = (self.output[name.toString()]! as? MLMultiArray)!
		let l = output.count
		var v = RustVec<Float32>()
		output.withUnsafeMutableBytes { ptr, strides in
			let p = ptr.baseAddress!.assumingMemoryBound(to: Float32.self)
			v = rust_vec_from_ptr_f32(p, UInt(l))
		}
		return v
	}
	func outputI32(name: RustString) -> RustVec<Int32> {
		let output = (self.output[name.toString()]! as? MLMultiArray)!
		let l = output.count
		var v = RustVec<Int32>()
		output.withUnsafeMutableBytes { ptr, strides in
			let p = ptr.baseAddress!.assumingMemoryBound(to: Int32.self)
			v = rust_vec_from_ptr_i32(p, UInt(l))
		}
		return v
	}
	func outputU16(name: RustString) -> RustVec<UInt16> {
		let output = (self.output[name.toString()]! as? MLMultiArray)!
		let l = output.count
		var v = RustVec<UInt16>()
		output.withUnsafeMutableBytes { ptr, strides in
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
			return ()
		})
	let m = Model.init()
	m.modelCompiledAsset = try! MLModelAsset.init(specification: data)
	m.computeUnits = computeUnits
	return m
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
		compiledPath = try! MLModel.compileModel(at: url)
	}
	let m = Model.init()
	m.compiledPath = compiledPath
	m.computeUnits = computeUnits
	return m
}

class Model: @unchecked Sendable {
	var compiledPath: URL? = nil
	var modelCompiledAsset: MLModelAsset? = nil
	var model: MLModel? = nil
	var dict: [String: Any] = [:]
	var inputs: MLDictionaryFeatureProvider? = nil
	var outputs: [String: Any] = [:]
	var computeUnits: MLComputeUnits = .cpuAndNeuralEngine

	init() {}

	func load() {
		let config = MLModelConfiguration.init()
		config.computeUnits = self.computeUnits
		if self.compiledPath == nil {
			let semaphore = DispatchSemaphore(value: 0)
			Task { [weak self] in
				guard let self else { return }
				let asset = self.modelCompiledAsset!
				let res = try! await MLModel.load(asset: asset, configuration: config)
				self.model = res
				semaphore.signal()
			}
			semaphore.wait()
		} else {
			let loadedModel = try! MLModel(contentsOf: self.compiledPath!, configuration: config)
			self.model = loadedModel
		}
	}

	func unload() {
		self.model = nil
	}

	func description() -> ModelDescription {
		return ModelDescription(desc: self.model!.modelDescription)
	}

	func bindOutputF32(
		shape: RustVec<Int32>, featureName: RustString, data: UnsafeMutablePointer<Float32>,
		len: UInt
	) {
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
		} catch {
			print("Unexpected error; \(error)")
		}
	}

	func predict() -> ModelOutput {
		do {
			self.inputs = try MLDictionaryFeatureProvider.init(dictionary: self.dict)
			let opts = MLPredictionOptions.init()
			opts.outputBackings = self.outputs
			try self.model!.prediction(from: self.inputs!, options: opts)
			let outputs = self.outputs
			self.outputs = [:]
			return ModelOutput(output: outputs)
		} catch {
			print("Unexpected error: \(error)")
			return ModelOutput(output: self.outputs)
		}
	}

	func bindInputF32(
		shape: RustVec<Int32>, featureName: RustString, data: UnsafeMutablePointer<Float32>,
		len: UInt
	) {
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
		} catch {
			print("Unexpected error; \(error)")
		}
	}

	func bindInputI32(
		shape: RustVec<Int32>, featureName: RustString, data: UnsafeMutablePointer<Int32>, len: UInt
	) {
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
		} catch {
			print("Unexpected error; \(error)")
		}
	}

	func bindInputU16(
		shape: RustVec<Int32>, featureName: RustString, data: UnsafeMutablePointer<UInt16>,
		len: UInt
	) {
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
		} catch {
			print("Unexpected error; \(error)")
		}
	}
}
