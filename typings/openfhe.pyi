class BINFHE_METHOD:
	AP: Any
	GINX: Any
	INVALID_METHOD: Any
	LMKCDEY: Any
	name: Any
	value: Any

	...
class BINFHE_OUTPUT:
	BOOTSTRAPPED: Any
	FRESH: Any
	INVALID_OUTPUT: Any
	name: Any
	value: Any

	...
class BINFHE_PARAMSET:
	MEDIUM: Any
	SIGNED_MOD_TEST: Any
	STD128: Any
	STD128Q: Any
	STD128Q_3: Any
	STD128Q_3_LMKCDEY: Any
	STD128Q_4: Any
	STD128Q_4_LMKCDEY: Any
	STD128Q_LMKCDEY: Any
	STD128_3: Any
	STD128_3_LMKCDEY: Any
	STD128_4: Any
	STD128_4_LMKCDEY: Any
	STD128_AP: Any
	STD128_LMKCDEY: Any
	STD192: Any
	STD192Q: Any
	STD192Q_3: Any
	STD192Q_4: Any
	STD256: Any
	STD256Q: Any
	STD256Q_3: Any
	STD256Q_4: Any
	TOY: Any
	name: Any
	value: Any

	...
class BINGATE:
	AND: Any
	NAND: Any
	NOR: Any
	OR: Any
	XNOR: Any
	XNOR_FAST: Any
	XOR: Any
	XOR_FAST: Any
	name: Any
	value: Any

	...
class BinFHEContext:
	BTKeyGen: Any
	Bootstrap: Any
	ClearBTKeys: Any
	Decrypt: Any
	Encrypt: Any
	EvalBinGate: Any
	EvalConstant: Any
	EvalDecomp: Any
	EvalFloor: Any
	EvalFunc: Any
	EvalNOT: Any
	EvalSign: Any
	GenerateBinFHEContext: Any
	GenerateLUTviaFunction: Any
	GetBeta: Any
	GetBinFHEScheme: Any
	GetLWEScheme: Any
	GetMaxPlaintextSpace: Any
	GetParams: Any
	GetPublicKey: Any
	GetRefreshKey: Any
	GetSwitchKey: Any
	Getn: Any
	Getq: Any
	KeyGen: Any
	KeyGenN: Any
	KeyGenPair: Any
	LoadBinary: Any
	LoadJSON: Any
	LoadPortableBinary: Any
	SaveBinary: Any
	SaveJSON: Any
	SavePortableBinary: Any
	SerializedObjectName: Any
	SerializedVersion: Any

	...
class CCParamsBFVRNS:
	GetBatchSize: Any
	GetCKKSDataType: Any
	GetCompositeDegree: Any
	GetDecryptionNoiseMode: Any
	GetDesiredPrecision: Any
	GetDigitSize: Any
	GetEncryptionTechnique: Any
	GetEvalAddCount: Any
	GetExecutionMode: Any
	GetFirstModSize: Any
	GetInteractiveBootCompressionLevel: Any
	GetKeySwitchCount: Any
	GetKeySwitchTechnique: Any
	GetMaxRelinSkDeg: Any
	GetMultipartyMode: Any
	GetMultiplicationTechnique: Any
	GetMultiplicativeDepth: Any
	GetNoiseEstimate: Any
	GetNumAdversarialQueries: Any
	GetNumLargeDigits: Any
	GetPREMode: Any
	GetPRENumHops: Any
	GetPlaintextModulus: Any
	GetRegisterWordSize: Any
	GetRingDim: Any
	GetScalingModSize: Any
	GetScalingTechnique: Any
	GetScheme: Any
	GetSecretKeyDist: Any
	GetSecurityLevel: Any
	GetStandardDeviation: Any
	GetStatisticalSecurity: Any
	SetBatchSize: Any
	SetCKKSDataType: Any
	SetCompositeDegree: Any
	SetDecryptionNoiseMode: Any
	SetDesiredPrecision: Any
	SetDigitSize: Any
	SetEncryptionTechnique: Any
	SetEvalAddCount: Any
	SetExecutionMode: Any
	SetFirstModSize: Any
	SetInteractiveBootCompressionLevel: Any
	SetKeySwitchCount: Any
	SetKeySwitchTechnique: Any
	SetMaxRelinSkDeg: Any
	SetMultipartyMode: Any
	SetMultiplicationTechnique: Any
	SetMultiplicativeDepth: Any
	SetNoiseEstimate: Any
	SetNumAdversarialQueries: Any
	SetNumLargeDigits: Any
	SetPREMode: Any
	SetPRENumHops: Any
	SetPlaintextModulus: Any
	SetRegisterWordSize: Any
	SetRingDim: Any
	SetScalingModSize: Any
	SetScalingTechnique: Any
	SetSecretKeyDist: Any
	SetSecurityLevel: Any
	SetStandardDeviation: Any
	SetStatisticalSecurity: Any
	SetThresholdNumOfParties: Any

	...
class CCParamsBGVRNS:
	GetBatchSize: Any
	GetCKKSDataType: Any
	GetCompositeDegree: Any
	GetDecryptionNoiseMode: Any
	GetDesiredPrecision: Any
	GetDigitSize: Any
	GetEncryptionTechnique: Any
	GetEvalAddCount: Any
	GetExecutionMode: Any
	GetFirstModSize: Any
	GetInteractiveBootCompressionLevel: Any
	GetKeySwitchCount: Any
	GetKeySwitchTechnique: Any
	GetMaxRelinSkDeg: Any
	GetMultipartyMode: Any
	GetMultiplicationTechnique: Any
	GetMultiplicativeDepth: Any
	GetNoiseEstimate: Any
	GetNumAdversarialQueries: Any
	GetNumLargeDigits: Any
	GetPREMode: Any
	GetPRENumHops: Any
	GetPlaintextModulus: Any
	GetRegisterWordSize: Any
	GetRingDim: Any
	GetScalingModSize: Any
	GetScalingTechnique: Any
	GetScheme: Any
	GetSecretKeyDist: Any
	GetSecurityLevel: Any
	GetStandardDeviation: Any
	GetStatisticalSecurity: Any
	SetBatchSize: Any
	SetCKKSDataType: Any
	SetCompositeDegree: Any
	SetDecryptionNoiseMode: Any
	SetDesiredPrecision: Any
	SetDigitSize: Any
	SetEncryptionTechnique: Any
	SetEvalAddCount: Any
	SetExecutionMode: Any
	SetFirstModSize: Any
	SetInteractiveBootCompressionLevel: Any
	SetKeySwitchCount: Any
	SetKeySwitchTechnique: Any
	SetMaxRelinSkDeg: Any
	SetMultipartyMode: Any
	SetMultiplicationTechnique: Any
	SetMultiplicativeDepth: Any
	SetNoiseEstimate: Any
	SetNumAdversarialQueries: Any
	SetNumLargeDigits: Any
	SetPREMode: Any
	SetPRENumHops: Any
	SetPlaintextModulus: Any
	SetRegisterWordSize: Any
	SetRingDim: Any
	SetScalingModSize: Any
	SetScalingTechnique: Any
	SetSecretKeyDist: Any
	SetSecurityLevel: Any
	SetStandardDeviation: Any
	SetStatisticalSecurity: Any
	SetThresholdNumOfParties: Any

	...
class CCParamsCKKSRNS:
	GetBatchSize: Any
	GetCKKSDataType: Any
	GetCompositeDegree: Any
	GetDecryptionNoiseMode: Any
	GetDesiredPrecision: Any
	GetDigitSize: Any
	GetEncryptionTechnique: Any
	GetEvalAddCount: Any
	GetExecutionMode: Any
	GetFirstModSize: Any
	GetInteractiveBootCompressionLevel: Any
	GetKeySwitchCount: Any
	GetKeySwitchTechnique: Any
	GetMaxRelinSkDeg: Any
	GetMultipartyMode: Any
	GetMultiplicationTechnique: Any
	GetMultiplicativeDepth: Any
	GetNoiseEstimate: Any
	GetNumAdversarialQueries: Any
	GetNumLargeDigits: Any
	GetPREMode: Any
	GetPRENumHops: Any
	GetPlaintextModulus: Any
	GetRegisterWordSize: Any
	GetRingDim: Any
	GetScalingModSize: Any
	GetScalingTechnique: Any
	GetScheme: Any
	GetSecretKeyDist: Any
	GetSecurityLevel: Any
	GetStandardDeviation: Any
	GetStatisticalSecurity: Any
	SetBatchSize: Any
	SetCKKSDataType: Any
	SetCompositeDegree: Any
	SetDecryptionNoiseMode: Any
	SetDesiredPrecision: Any
	SetDigitSize: Any
	SetEncryptionTechnique: Any
	SetEvalAddCount: Any
	SetExecutionMode: Any
	SetFirstModSize: Any
	SetInteractiveBootCompressionLevel: Any
	SetKeySwitchCount: Any
	SetKeySwitchTechnique: Any
	SetMaxRelinSkDeg: Any
	SetMultipartyMode: Any
	SetMultiplicationTechnique: Any
	SetMultiplicativeDepth: Any
	SetNoiseEstimate: Any
	SetNumAdversarialQueries: Any
	SetNumLargeDigits: Any
	SetPREMode: Any
	SetPRENumHops: Any
	SetPlaintextModulus: Any
	SetRegisterWordSize: Any
	SetRingDim: Any
	SetScalingModSize: Any
	SetScalingTechnique: Any
	SetSecretKeyDist: Any
	SetSecurityLevel: Any
	SetStandardDeviation: Any
	SetStatisticalSecurity: Any
	SetThresholdNumOfParties: Any

	...
class CKKSDataType:
	COMPLEX: Any
	REAL: Any
	name: Any
	value: Any

	...
class COMPRESSION_LEVEL:
	COMPACT: Any
	SLACK: Any
	name: Any
	value: Any

	...
class Ciphertext:
	Clone: Any
	GetCryptoContext: Any
	GetElements: Any
	GetElementsMutable: Any
	GetEncodingType: Any
	GetLevel: Any
	GetNoiseScaleDeg: Any
	GetSlots: Any
	RemoveElement: Any
	SetElements: Any
	SetElementsMove: Any
	SetLevel: Any
	SetNoiseScaleDeg: Any
	SetSlots: Any

	...
ClearEvalMultKeys: builtin_function_or_method
class CryptoContext:
	ClearEvalAutomorphismKeys: Any
	Compress: Any
	Decrypt: Any
	DeserializeEvalAutomorphismKey: Any
	DeserializeEvalMultKey: Any
	Enable: Any
	Encrypt: Any
	EvalAdd: Any
	EvalAddInPlace: Any
	EvalAddMany: Any
	EvalAddManyInPlace: Any
	EvalAddMutable: Any
	EvalAddMutableInPlace: Any
	EvalAtIndex: Any
	EvalAtIndexKeyGen: Any
	EvalAutomorphismKeyGen: Any
	EvalBootstrap: Any
	EvalBootstrapKeyGen: Any
	EvalBootstrapSetup: Any
	EvalCKKStoFHEW: Any
	EvalCKKStoFHEWKeyGen: Any
	EvalCKKStoFHEWPrecompute: Any
	EvalCKKStoFHEWSetup: Any
	EvalChebyshevFunction: Any
	EvalChebyshevSeries: Any
	EvalChebyshevSeriesLinear: Any
	EvalChebyshevSeriesPS: Any
	EvalCompareSchemeSwitching: Any
	EvalCompareSwitchPrecompute: Any
	EvalCos: Any
	EvalDivide: Any
	EvalFHEWtoCKKS: Any
	EvalFHEWtoCKKSKeyGen: Any
	EvalFHEWtoCKKSSetup: Any
	EvalFastRotation: Any
	EvalFastRotationExt: Any
	EvalFastRotationPrecompute: Any
	EvalInnerProduct: Any
	EvalLinearWSum: Any
	EvalLinearWSumMutable: Any
	EvalLogistic: Any
	EvalMaxSchemeSwitching: Any
	EvalMaxSchemeSwitchingAlt: Any
	EvalMerge: Any
	EvalMinSchemeSwitching: Any
	EvalMinSchemeSwitchingAlt: Any
	EvalMult: Any
	EvalMultAndRelinearize: Any
	EvalMultKeyGen: Any
	EvalMultKeysGen: Any
	EvalMultMany: Any
	EvalMultMutable: Any
	EvalMultMutableInPlace: Any
	EvalMultNoRelin: Any
	EvalNegate: Any
	EvalNegateInPlace: Any
	EvalPoly: Any
	EvalPolyLinear: Any
	EvalPolyPS: Any
	EvalRotate: Any
	EvalRotateKeyGen: Any
	EvalSchemeSwitchingKeyGen: Any
	EvalSchemeSwitchingSetup: Any
	EvalSin: Any
	EvalSquare: Any
	EvalSquareInPlace: Any
	EvalSquareMutable: Any
	EvalSub: Any
	EvalSubInPlace: Any
	EvalSubMutable: Any
	EvalSubMutableInPlace: Any
	EvalSum: Any
	EvalSumCols: Any
	EvalSumColsKeyGen: Any
	EvalSumKeyGen: Any
	EvalSumRows: Any
	EvalSumRowsKeyGen: Any
	FindAutomorphismIndex: Any
	FindAutomorphismIndices: Any
	GetBatchSize: Any
	GetBinCCForSchemeSwitch: Any
	GetCKKSDataType: Any
	GetCompositeDegree: Any
	GetCyclotomicOrder: Any
	GetDigitSize: Any
	GetEvalAddCount: Any
	GetEvalAutomorphismKeyMap: Any
	GetEvalMultKeyVector: Any
	GetEvalSumKeyMap: Any
	GetKeyGenLevel: Any
	GetKeySwitchCount: Any
	GetModulus: Any
	GetModulusCKKS: Any
	GetMultiplicativeDepth: Any
	GetNoiseEstimate: Any
	GetPRENumHops: Any
	GetPlaintextModulus: Any
	GetRegisterWordSize: Any
	GetRingDimension: Any
	GetScalingFactorReal: Any
	GetScalingTechnique: Any
	InsertEvalAutomorphismKey: Any
	InsertEvalMultKey: Any
	InsertEvalSumKey: Any
	IntBootAdd: Any
	IntBootAdjustScale: Any
	IntBootDecrypt: Any
	IntBootEncrypt: Any
	IntMPBootAdd: Any
	IntMPBootAdjustScale: Any
	IntMPBootDecrypt: Any
	IntMPBootEncrypt: Any
	IntMPBootRandomElementGen: Any
	KeyGen: Any
	KeySwitchGen: Any
	MakeCKKSPackedPlaintext: Any
	MakeCoefPackedPlaintext: Any
	MakePackedPlaintext: Any
	MakeStringPlaintext: Any
	ModReduce: Any
	ModReduceInPlace: Any
	MultiAddEvalAutomorphismKeys: Any
	MultiAddEvalKeys: Any
	MultiAddEvalMultKeys: Any
	MultiAddEvalSumKeys: Any
	MultiAddPubKeys: Any
	MultiEvalAtIndexKeyGen: Any
	MultiEvalSumKeyGen: Any
	MultiKeySwitchGen: Any
	MultiMultEvalKey: Any
	MultipartyDecryptFusion: Any
	MultipartyDecryptLead: Any
	MultipartyDecryptMain: Any
	MultipartyKeyGen: Any
	ReEncrypt: Any
	ReKeyGen: Any
	Relinearize: Any
	RelinearizeInPlace: Any
	Rescale: Any
	RescaleInPlace: Any
	SerializeEvalAutomorphismKey: Any
	SerializeEvalMultKey: Any
	SetEvalAddCount: Any
	SetKeyGenLevel: Any
	SetKeySwitchCount: Any
	SetMultiplicativeDepth: Any
	SetNoiseEstimate: Any
	SetPRENumHops: Any
	get_ptr: Any

	...
class DCRTPoly:

	...
class DecryptionNoiseMode:
	FIXED_NOISE_DECRYPT: Any
	NOISE_FLOODING_DECRYPT: Any
	name: Any
	value: Any

	...
DeserializeCiphertext: builtin_function_or_method
DeserializeCiphertextString: builtin_function_or_method
DeserializeCryptoContext: builtin_function_or_method
DeserializeCryptoContextString: builtin_function_or_method
DeserializeEvalAutomorphismKeyString: builtin_function_or_method
DeserializeEvalKey: builtin_function_or_method
DeserializeEvalKeyMap: builtin_function_or_method
DeserializeEvalKeyMapString: builtin_function_or_method
DeserializeEvalKeyMapVectorString: builtin_function_or_method
DeserializeEvalKeyString: builtin_function_or_method
DeserializeEvalMultKeyString: builtin_function_or_method
DeserializePrivateKey: builtin_function_or_method
DeserializePrivateKeyString: builtin_function_or_method
DeserializePublicKey: builtin_function_or_method
DeserializePublicKeyString: builtin_function_or_method
DisablePrecomputeCRTTablesAfterDeserializaton: builtin_function_or_method
EnablePrecomputeCRTTablesAfterDeserializaton: builtin_function_or_method
class EncryptionTechnique:
	EXTENDED: Any
	STANDARD: Any
	name: Any
	value: Any

	...
class EvalKey:
	GetKeyTag: Any
	SetKeyTag: Any

	...
class EvalKeyMap:

	...
class ExecutionMode:
	EXEC_EVALUATION: Any
	EXEC_NOISE_ESTIMATION: Any
	name: Any
	value: Any

	...
class FHECKKSRNS:
	GetBootstrapDepth: Any

	...
class Format:
	COEFFICIENT: Any
	EVALUATION: Any
	name: Any
	value: Any

	...
GenCryptoContext: builtin_function_or_method
GetAllContexts: builtin_function_or_method
class KEYGEN_MODE:
	PUB_ENCRYPT: Any
	SYM_ENCRYPT: Any
	name: Any
	value: Any

	...
class KeyPair:
	good: Any
	publicKey: Any
	secretKey: Any

	...
class KeySwitchTechnique:
	BV: Any
	HYBRID: Any
	INVALID_KS_TECH: Any
	name: Any
	value: Any

	...
class LWECiphertext:
	GetLength: Any
	GetModulus: Any

	...
class LWEPrivateKey:
	GetLength: Any

	...
class MultipartyMode:
	FIXED_NOISE_MULTIPARTY: Any
	INVALID_MULTIPARTY_MODE: Any
	NOISE_FLOODING_MULTIPARTY: Any
	name: Any
	value: Any

	...
class MultiplicationTechnique:
	BEHZ: Any
	HPS: Any
	HPSPOVERQ: Any
	HPSPOVERQLEVELED: Any
	name: Any
	value: Any

	...
class PKESchemeFeature:
	ADVANCEDSHE: Any
	FHE: Any
	KEYSWITCH: Any
	LEVELEDSHE: Any
	MULTIPARTY: Any
	PKE: Any
	PRE: Any
	SCHEMESWITCH: Any
	name: Any
	value: Any

	...
class ParmType:

	...
class Plaintext:
	Decode: Any
	Encode: Any
	GetCKKSPackedValue: Any
	GetCoefPackedValue: Any
	GetFormattedValues: Any
	GetLength: Any
	GetLevel: Any
	GetLogError: Any
	GetLogPrecision: Any
	GetNoiseScaleDeg: Any
	GetPackedValue: Any
	GetRealPackedValue: Any
	GetScalingFactor: Any
	GetSchemeID: Any
	GetSlots: Any
	GetStringValue: Any
	HighBound: Any
	IsEncoded: Any
	LowBound: Any
	SetFormat: Any
	SetIntVectorValue: Any
	SetLength: Any
	SetLevel: Any
	SetNoiseScaleDeg: Any
	SetScalingFactor: Any
	SetSlots: Any
	SetStringValue: Any

	...
class PrivateKey:
	GetCryptoContext: Any
	GetKeyTag: Any
	SetKeyTag: Any

	...
class ProxyReEncryptionMode:
	FIXED_NOISE_HRA: Any
	INDCPA: Any
	NOISE_FLOODING_HRA: Any
	NOT_SET: Any
	name: Any
	value: Any

	...
class PublicKey:
	GetKeyTag: Any
	SetKeyTag: Any

	...
ReleaseAllContexts: builtin_function_or_method
class SCHEME:
	BFVRNS_SCHEME: Any
	BGVRNS_SCHEME: Any
	CKKSRNS_SCHEME: Any
	INVALID_SCHEME: Any
	name: Any
	value: Any

	...
class SERBINARY:

	...
class SERJSON:

	...
class ScalingTechnique:
	COMPOSITESCALINGAUTO: Any
	COMPOSITESCALINGMANUAL: Any
	FIXEDAUTO: Any
	FIXEDMANUAL: Any
	FLEXIBLEAUTO: Any
	FLEXIBLEAUTOEXT: Any
	INVALID_RS_TECHNIQUE: Any
	NORESCALE: Any
	name: Any
	value: Any

	...
class SchSwchParams:
	GetArbitraryFunctionEvaluation: Any
	GetBStepLTrCKKStoFHEW: Any
	GetBStepLTrFHEWtoCKKS: Any
	GetBatchSize: Any
	GetComputeArgmin: Any
	GetCtxtModSizeFHEWIntermedSwch: Any
	GetCtxtModSizeFHEWLargePrec: Any
	GetInitialCKKSModulus: Any
	GetLevelLTrCKKStoFHEW: Any
	GetLevelLTrFHEWtoCKKS: Any
	GetNumSlotsCKKS: Any
	GetNumValues: Any
	GetOneHotEncoding: Any
	GetRingDimension: Any
	GetScalingModSize: Any
	GetSecurityLevelCKKS: Any
	GetSecurityLevelFHEW: Any
	GetUseAltArgmin: Any
	GetUseDynamicModeFHEW: Any
	SetArbitraryFunctionEvaluation: Any
	SetBStepLTrCKKStoFHEW: Any
	SetBStepLTrFHEWtoCKKS: Any
	SetBatchSize: Any
	SetComputeArgmin: Any
	SetCtxtModSizeFHEWIntermedSwch: Any
	SetCtxtModSizeFHEWLargePrec: Any
	SetInitialCKKSModulus: Any
	SetLevelLTrCKKStoFHEW: Any
	SetLevelLTrFHEWtoCKKS: Any
	SetNumSlotsCKKS: Any
	SetNumValues: Any
	SetOneHotEncoding: Any
	SetRingDimension: Any
	SetScalingModSize: Any
	SetSecurityLevelCKKS: Any
	SetSecurityLevelFHEW: Any
	SetUseAltArgmin: Any
	SetUseDynamicModeFHEW: Any

	...
class SecretKeyDist:
	GAUSSIAN: Any
	SPARSE_TERNARY: Any
	UNIFORM_TERNARY: Any
	name: Any
	value: Any

	...
class SecurityLevel:
	HEStd_128_classic: Any
	HEStd_192_classic: Any
	HEStd_256_classic: Any
	HEStd_NotSet: Any
	name: Any
	value: Any

	...
Serialize: builtin_function_or_method
SerializeEvalAutomorphismKeyString: builtin_function_or_method
SerializeEvalMultKeyString: builtin_function_or_method
SerializeToFile: builtin_function_or_method
get_native_int: builtin_function_or_method
