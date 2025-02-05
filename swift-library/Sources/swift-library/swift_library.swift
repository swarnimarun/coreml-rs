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
}

class ModelOutput {
	var output: MLFeatureProvider? = nil
	init(output: MLFeatureProvider?) {
		self.output = output
	}
	func outputDescription() -> RustVec<RustString> {
		let ret = RustVec<RustString>()
		for key in self.output!.featureNames {
			let str = "\(key):\(self.output!.featureValue(for: key)!)".intoRustString()
			ret.push(value: str)
		}
		return ret
	}
	func bytesFrom(name: RustString) -> RustVec<Float32> {
		let output = self.output!.featureValue(for: name.toString())!.multiArrayValue!
		let l = output.count
		var v = RustVec<Float32>()
		output.withUnsafeMutableBytes { ptr, strides in
			let p = ptr.baseAddress!.assumingMemoryBound(to: Float32.self)
			v = rust_vec_from_ptr(p, UInt(l))
		}
		return v
	}
}

class Model {
	var compiledPath: URL? = nil
	var model: MLModel? = nil
	var dict: [String: Any] = [:]
	var inputs: MLDictionaryFeatureProvider? = nil

	init(path: RustString) {
		let url = URL(string: path.toString())!
		self.compiledPath = try! MLModel.compileModel(at: url)
	}

	func load() {
		let loadedModel = try! MLModel(
			contentsOf: self.compiledPath!, configuration: MLModelConfiguration.init())
		self.model = loadedModel
	}

	func description() -> ModelDescription {
		return ModelDescription(desc: self.model!.modelDescription)
	}

	func predict() -> ModelOutput {
		do {
			self.inputs = try MLDictionaryFeatureProvider.init(dictionary: self.dict)
			let res = try self.model!.prediction(from: self.inputs!)
			return ModelOutput(output: res)
		} catch {
			print("Unexpected error: \(error)")
			return ModelOutput(output: nil)
		}
	}

	func bindInput(shape: RustVec<Int32>, featureName: RustString, data: RustVec<Float32>) {
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
			// let array = try MLMultiArray.init(dataPointer: shape.ptr, shape: arr, dataType: MLMultiArrayDataType.float32, strides: stride)
			let array = try MLMultiArray.init(shape: arr, dataType: MLMultiArrayDataType.float32)
			let value = MLFeatureValue(multiArray: array)
			self.dict[featureName.toString()] = value
		} catch {
			print("Unexpected error; \(error)")
		}

	}
}
