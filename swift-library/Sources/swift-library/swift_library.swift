import CoreML

// Allocate Core ML-owned storage. No Rust allocation is retained or freed by Swift.
private func copiedMultiArray(
	shape: [NSNumber], values: [NSNumber], dataType: MLMultiArrayDataType
) throws -> MLMultiArray {
	guard !shape.isEmpty else { throw RuntimeError("empty tensor shape") }
	var count = 1
	for dimension in shape {
		guard dimension.int64Value > 0, dimension.uint64Value <= UInt64(Int.max) else {
			throw RuntimeError("invalid tensor dimension")
		}
		let product = count.multipliedReportingOverflow(by: dimension.intValue)
		guard !product.overflow else { throw RuntimeError("tensor shape overflow") }
		count = product.partialValue
	}
	guard count == values.count else { throw RuntimeError("tensor length does not match shape") }
	let array = try MLMultiArray(shape: shape, dataType: dataType)
	for (index, value) in values.enumerated() {
		array[index] = value
	}
	return array
}


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

// Only the detached task writes the result. The semaphore publishes it to the
// calling thread, which does not read it until completion. This operation never
// shares a Model or BatchModel with the task.
private final class AssetLoadOperation: @unchecked Sendable {
	let asset: MLModelAsset
	let configuration: MLModelConfiguration
	private var result: MLModel? = nil

	init(asset: MLModelAsset, configuration: MLModelConfiguration) {
		self.asset = asset
		self.configuration = configuration
	}

	func run() -> MLModel? {
		let completed = DispatchSemaphore(value: 0)
		Task.detached { [self] in
			defer { completed.signal() }
			do {
				result = try await MLModel.load(asset: asset, configuration: configuration)
			} catch {
				result = nil
			}
		}
		completed.wait()
		return result
	}
}

class BatchModel: @unchecked Sendable {
	var compiledPath: URL? = nil
	var model: MLModel? = nil
	var modelCompiledAsset: MLModelAsset? = nil
	// Retain the specification for the full asset/model lifetime.
	var specificationData: Data? = nil
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
		// IMPORTANT: disable experimentalMLE5EngineUsage, seems to cause crash on newer MacOS versions
		config.setValue(1, forKey: "experimentalMLE5EngineUsage")
		do {
			if self.compiledPath == nil {
				guard let asset = self.modelCompiledAsset else { return false }
				self.model = AssetLoadOperation(asset: asset, configuration: config).run()
				return self.model != nil
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
		shape: RustVec<UInt>, featureName: RustString, data: RustVec<Float32>, idx: Int
	) -> Bool {
		// Only existing entries or the next entry may be bound; never index a gap.
		guard idx >= 0, idx <= self.inputs.count else { return false }
		do {
			let array = try copiedMultiArray(
				shape: shape.map { NSNumber(value: $0) },
				values: data.map { NSNumber(value: $0) }, dataType: .float32)
			if idx == self.inputs.count { self.inputs.append(BatchModelInput()) }
			self.inputs[idx].dict[featureName.toString()] = MLFeatureValue(multiArray: array)
			return true
		} catch {
			print("Unexpected input error: \(error)")
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
	private func outputArray(name: String) -> MLMultiArray? {
		guard !hasFailedToLoad(), let value = self.output?[name] else { return nil }
		if let feature = value as? MLFeatureValue { return feature.multiArrayValue }
		return value as? MLMultiArray
	}

	func outputF32(name: RustString) -> RustVec<Float32> {
		let result = RustVec<Float32>()
		guard let out = outputArray(name: name.toString()), out.dataType == .float32 else {
			return result
		}
		for index in 0..<out.count {
			result.push(value: out[index].floatValue)
		}
		return result
	}
	func outputI32(name: RustString) -> RustVec<Int32> {
		let result = RustVec<Int32>()
		guard let out = outputArray(name: name.toString()), out.dataType == .int32 else {
			return result
		}
		for index in 0..<out.count {
			result.push(value: out[index].int32Value)
		}
		return result
	}
	func outputU16(name: RustString) -> RustVec<UInt16> {
		let result = RustVec<UInt16>()
		guard let out = outputArray(name: name.toString()), out.dataType == .float16 else {
			return result
		}
		for index in 0..<out.count {
			result.push(value: Float16(out[index].floatValue).bitPattern)
		}
		return result
	}
}

func initWithCompiledAsset(
	bytes: RustVec<UInt8>, compute: ComputePlatform
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
	// RustVec owns its allocation; Data takes an independent copy before it drops.
	let data = Data(Array(bytes))
	do {
		let m = Model.init(failedToLoad: false)
		m.specificationData = data
		m.modelCompiledAsset = try MLModelAsset.init(specification: data)
		m.computeUnits = computeUnits
		return m
	} catch {
		let m = Model.init(failedToLoad: true)
		return m
	}
}

func initWithCompiledAssetBatch(
	bytes: RustVec<UInt8>, compute: ComputePlatform
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
	// RustVec owns its allocation; Data takes an independent copy before it drops.
	let data = Data(Array(bytes))
	do {
		let m = BatchModel.init(failedToLoad: false)
		m.specificationData = data
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
		compiledPath = urlFromPath(path.toString())
	} else {
		let url = urlFromPath(path.toString())
		do {
			compiledPath = try MLModel.compileModel(at: url)
		} catch {
			print("Failed to compile CoreML model at \(url): \(error)")
			return Model.init(failedToLoad: true)
		}
	}
	let m = Model.init(failedToLoad: false)
	m.compiledPath = compiledPath
	m.computeUnits = computeUnits
	return m
}

// Compile model and overwrite the file to the permanent location, replacing it if necessary
func compileToPath(model: RustString, to: RustString, name: RustString) -> Bool {
	let url = URL(string: model.toString())!
	do {
		let compiledPath = try MLModel.compileModel(at: url)
		let fileManager = FileManager.default
		let appSupportURL = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
		print(appSupportURL)
		let permanentURL = appSupportURL.appendingPathComponent(name.toString())
		_ = try fileManager.replaceItemAt(permanentURL, withItemAt: compiledPath)
	} catch {
		return false
	}
	return true
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
		compiledPath = urlFromPath(path.toString())
	} else {
		let url = urlFromPath(path.toString())
		do {
			compiledPath = try MLModel.compileModel(at: url)
		} catch {
			print("Failed to compile CoreML batch model at \(url): \(error)")
			return BatchModel.init(failedToLoad: true)
		}
	}
	let m = BatchModel.init(failedToLoad: false)
	m.compiledPath = compiledPath
	m.computeUnits = computeUnits
	return m
}

func urlFromPath(_ path: String) -> URL {
	if path.hasPrefix("file://") {
		return URL(string: path)!
	}
	return URL(fileURLWithPath: path)
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
	// Retain the specification for the full asset/model lifetime.
	var specificationData: Data? = nil
	var model: MLModel? = nil
	var dict: [String: Any] = [:]
	var outputs: [String: Any] = [:]
	var computeUnits: MLComputeUnits = .cpuAndNeuralEngine
	var disableExperimentalMLE: Bool = false

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

	func setDisableExperimentalMLE(disabled: Bool) {
		self.disableExperimentalMLE = disabled
	}

	func load() -> Bool {
		if hasFailedToLoad() { return false }
		let config = MLModelConfiguration.init()
		config.computeUnits = self.computeUnits
		if self.disableExperimentalMLE {
			config.setValue(1, forKey: "experimentalMLE5EngineUsage")
		}
		do {
			if self.compiledPath == nil {
				guard let asset = self.modelCompiledAsset else { return false }
				self.model = AssetLoadOperation(asset: asset, configuration: config).run()
				return self.model != nil
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
		shape: RustVec<Int32>, featureName: RustString, data: RustVec<Float32>
	) -> Bool {
		if hasFailedToLoad() { return false }
		do {
			let array = try copiedMultiArray(
				shape: shape.map { NSNumber(value: $0) },
				values: data.map { NSNumber(value: $0) }, dataType: .float32)
			self.outputs[featureName.toString()] = array
			return true
		} catch {
			print("Unexpected tensor binding error: \(error)")
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
			guard let model = self.model else {
				return ModelOutput(output: nil, error: RuntimeError("model is not loaded"))
			}
			let prediction = try model.prediction(from: input, options: opts)
			let outputs = (prediction as? MLDictionaryFeatureProvider)?.dictionary
			self.outputs = [:]
			self.dict = [:]
			return ModelOutput(output: outputs, error: nil, cpy: true)
		} catch {
			// print("Unexpected predict error: \(error)")
			return ModelOutput(output: nil, error: error)
		}
	}

	func bindInputF32(
		shape: RustVec<UInt>, featureName: RustString, data: RustVec<Float32>
	) -> Bool {
		do {
			let array = try copiedMultiArray(
				shape: shape.map { NSNumber(value: $0) },
				values: data.map { NSNumber(value: $0) }, dataType: .float32)
			self.dict[featureName.toString()] = MLFeatureValue(multiArray: array)
			return true
		} catch {
			print("Unexpected tensor binding error: \(error)")
			return false
		}
	}

	func bindInputI32(
		shape: RustVec<UInt>, featureName: RustString, data: RustVec<Int32>
	) -> Bool {
		do {
			let array = try copiedMultiArray(
				shape: shape.map { NSNumber(value: $0) },
				values: data.map { NSNumber(value: $0) }, dataType: .int32)
			self.dict[featureName.toString()] = MLFeatureValue(multiArray: array)
			return true
		} catch {
			print("Unexpected tensor binding error: \(error)")
			return false
		}
	}

	func bindInputU16(
		shape: RustVec<UInt>, featureName: RustString, data: RustVec<UInt16>
	) -> Bool {
		do {
			let array = try copiedMultiArray(
				shape: shape.map { NSNumber(value: $0) },
				values: data.map { NSNumber(value: Double(Float16(bitPattern: $0))) }, dataType: .float16)
			self.dict[featureName.toString()] = MLFeatureValue(multiArray: array)
			return true
		} catch {
			print("Unexpected tensor binding error: \(error)")
			return false
		}
	}
}
