import CoreML

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

class Model {
	var compiledPath: URL? = nil
	var model: MLModel? = nil
	var dict: [String: Any] = [:]
	var inputs: MLDictionaryFeatureProvider? = nil
	var outputs: [String: Any] = [:]
	var computeUnits: MLComputeUnits = .cpuAndNeuralEngine

	init(path: RustString, compute: ComputePlatform, compiled: Bool) {
		switch compute {
		case .Cpu:
			self.computeUnits = .cpuOnly
			break
		case .CpuAndANE:
			self.computeUnits = .cpuAndNeuralEngine
			break
		case .CpuAndGpu:
			self.computeUnits = .cpuAndGPU
			break
		}
		if compiled {
			self.compiledPath = URL(string: path.toString())!
		} else {
			let url = URL(string: path.toString())!
			self.compiledPath = try! MLModel.compileModel(at: url)
		}
	}

	func load() {
		let config = MLModelConfiguration.init()
		config.computeUnits = self.computeUnits
		let loadedModel = try! MLModel(contentsOf: self.compiledPath!, configuration: config)
		self.model = loadedModel
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
