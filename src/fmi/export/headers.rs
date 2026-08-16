// Vendored FMI 3.0 C declarations for the exported source FMU.
//
// A source FMU ships C that the importer compiles, so it must carry the FMI
// types and function prototypes. Rather than vendor the full upstream header
// tree, this is a single self-contained, ABI-accurate subset covering the
// Model-Exchange surface the generated wrapper implements. The types and
// function signatures match `fmi3PlatformTypes.h` / `fmi3FunctionTypes.h`
// (Modelica Association, 2-clause BSD) one-for-one, so the produced symbols are
// ABI-compatible with any FMI 3.0 importer (including this crate's own).
//
// `FMI3_Export` marks the entry points; on Windows it becomes
// `__declspec(dllexport)` so a DLL build exports them, and it is empty
// elsewhere (ELF/Mach-O export by default). The importer's own build sets the
// visibility it needs; this just makes the default DLL case work.

/// The vendored header, written to `sources/fmi3.h` in the FMU.
pub const FMI3_HEADER: &str = r#"#ifndef FASTSIM_FMI3_H
#define FASTSIM_FMI3_H

/* FMI 3.0 Model-Exchange C interface (self-contained ABI-accurate subset). */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- export marker ----------------------------------------------------- */
#if defined(_WIN32) || defined(__CYGWIN__)
#  define FMI3_Export __declspec(dllexport)
#else
#  define FMI3_Export
#endif

/* --- platform types (fmi3PlatformTypes.h) ------------------------------ */
typedef void*          fmi3Instance;
typedef void*          fmi3InstanceEnvironment;
typedef void*          fmi3FMUState;
typedef uint32_t       fmi3ValueReference;
typedef float          fmi3Float32;
typedef double         fmi3Float64;
typedef int8_t         fmi3Int8;
typedef uint8_t        fmi3UInt8;
typedef int16_t        fmi3Int16;
typedef uint16_t       fmi3UInt16;
typedef int32_t        fmi3Int32;
typedef uint32_t       fmi3UInt32;
typedef int64_t        fmi3Int64;
typedef uint64_t       fmi3UInt64;
typedef bool           fmi3Boolean;
typedef char           fmi3Char;
typedef const fmi3Char* fmi3String;
typedef uint8_t        fmi3Byte;

#define fmi3True  true
#define fmi3False false

/* --- status ------------------------------------------------------------ */
typedef enum {
    fmi3OK,
    fmi3Warning,
    fmi3Discard,
    fmi3Error,
    fmi3Fatal
} fmi3Status;

/* --- callbacks --------------------------------------------------------- */
typedef void (*fmi3LogMessageCallback)(fmi3InstanceEnvironment instanceEnvironment,
                                       fmi3Status status,
                                       fmi3String category,
                                       fmi3String message);

/* --- inquiry ----------------------------------------------------------- */
FMI3_Export const char* fmi3GetVersion(void);

/* --- creation / destruction ------------------------------------------- */
FMI3_Export fmi3Instance fmi3InstantiateModelExchange(
    fmi3String instanceName,
    fmi3String instantiationToken,
    fmi3String resourcePath,
    fmi3Boolean visible,
    fmi3Boolean loggingOn,
    fmi3InstanceEnvironment instanceEnvironment,
    fmi3LogMessageCallback logMessage);

FMI3_Export void fmi3FreeInstance(fmi3Instance instance);

/* --- initialization / termination ------------------------------------- */
FMI3_Export fmi3Status fmi3EnterInitializationMode(
    fmi3Instance instance,
    fmi3Boolean toleranceDefined,
    fmi3Float64 tolerance,
    fmi3Float64 startTime,
    fmi3Boolean stopTimeDefined,
    fmi3Float64 stopTime);

FMI3_Export fmi3Status fmi3ExitInitializationMode(fmi3Instance instance);
FMI3_Export fmi3Status fmi3EnterEventMode(fmi3Instance instance);
FMI3_Export fmi3Status fmi3Terminate(fmi3Instance instance);
FMI3_Export fmi3Status fmi3Reset(fmi3Instance instance);

FMI3_Export fmi3Status fmi3SetDebugLogging(
    fmi3Instance instance,
    fmi3Boolean loggingOn,
    size_t nCategories,
    const fmi3String categories[]);

/* --- discrete states (event mode) ------------------------------------- */
FMI3_Export fmi3Status fmi3UpdateDiscreteStates(
    fmi3Instance instance,
    fmi3Boolean* discreteStatesNeedUpdate,
    fmi3Boolean* terminateSimulation,
    fmi3Boolean* nominalsOfContinuousStatesChanged,
    fmi3Boolean* valuesOfContinuousStatesChanged,
    fmi3Boolean* nextEventTimeDefined,
    fmi3Float64* nextEventTime);

/* --- getters / setters ------------------------------------------------- */
FMI3_Export fmi3Status fmi3GetFloat64(
    fmi3Instance instance,
    const fmi3ValueReference valueReferences[],
    size_t nValueReferences,
    fmi3Float64 values[],
    size_t nValues);

FMI3_Export fmi3Status fmi3SetFloat64(
    fmi3Instance instance,
    const fmi3ValueReference valueReferences[],
    size_t nValueReferences,
    const fmi3Float64 values[],
    size_t nValues);

/* --- Model Exchange ---------------------------------------------------- */
FMI3_Export fmi3Status fmi3EnterContinuousTimeMode(fmi3Instance instance);

FMI3_Export fmi3Status fmi3CompletedIntegratorStep(
    fmi3Instance instance,
    fmi3Boolean noSetFMUStatePriorToCurrentPoint,
    fmi3Boolean* enterEventMode,
    fmi3Boolean* terminateSimulation);

FMI3_Export fmi3Status fmi3SetTime(fmi3Instance instance, fmi3Float64 time);

FMI3_Export fmi3Status fmi3SetContinuousStates(
    fmi3Instance instance,
    const fmi3Float64 continuousStates[],
    size_t nContinuousStates);

FMI3_Export fmi3Status fmi3GetContinuousStateDerivatives(
    fmi3Instance instance,
    fmi3Float64 derivatives[],
    size_t nContinuousStates);

FMI3_Export fmi3Status fmi3GetEventIndicators(
    fmi3Instance instance,
    fmi3Float64 eventIndicators[],
    size_t nEventIndicators);

FMI3_Export fmi3Status fmi3GetContinuousStates(
    fmi3Instance instance,
    fmi3Float64 continuousStates[],
    size_t nContinuousStates);

FMI3_Export fmi3Status fmi3GetNominalsOfContinuousStates(
    fmi3Instance instance,
    fmi3Float64 nominals[],
    size_t nContinuousStates);

FMI3_Export fmi3Status fmi3GetNumberOfContinuousStates(
    fmi3Instance instance,
    size_t* nContinuousStates);

FMI3_Export fmi3Status fmi3GetNumberOfEventIndicators(
    fmi3Instance instance,
    size_t* nEventIndicators);

/* --- directional derivatives (optional capability) --------------------- */
FMI3_Export fmi3Status fmi3GetDirectionalDerivative(
    fmi3Instance instance,
    const fmi3ValueReference unknowns[],
    size_t nUnknowns,
    const fmi3ValueReference knowns[],
    size_t nKnowns,
    const fmi3Float64 seed[],
    size_t nSeed,
    fmi3Float64 sensitivity[],
    size_t nSensitivity);

/* --- types used only by the refusing stubs below ----------------------- */
typedef const fmi3Byte* fmi3Binary;
typedef bool            fmi3Clock;

typedef enum {
    fmi3IntervalNotYetKnown,
    fmi3IntervalUnchanged,
    fmi3IntervalChanged
} fmi3IntervalQualifier;

typedef enum {
    fmi3Independent,
    fmi3Constant,
    fmi3Fixed,
    fmi3Tunable,
    fmi3Discrete,
    fmi3Dependent
} fmi3DependencyKind;

typedef void (*fmi3ClockUpdateCallback)(fmi3InstanceEnvironment instanceEnvironment);
typedef void (*fmi3IntermediateUpdateCallback)(
    fmi3InstanceEnvironment instanceEnvironment, fmi3Float64 intermediateUpdateTime,
    fmi3Boolean intermediateVariableSetRequested, fmi3Boolean intermediateVariableGetAllowed,
    fmi3Boolean intermediateStepFinished, fmi3Boolean canReturnEarly,
    fmi3Boolean* earlyReturnRequested, fmi3Float64* earlyReturnTime);
typedef void (*fmi3LockPreemptionCallback)(void);
typedef void (*fmi3UnlockPreemptionCallback)(void);

/* --- entry points this FMU does not implement -------------------------- */
/*
 * A Model-Exchange-only FMU still has to be *loadable* by an importer that
 * resolves the whole FMI 3.0 symbol table up front — the reference FMUs export
 * all of them, and FMPy, for one, resolves all of them eagerly no matter what
 * modelDescription.xml declares. Each of these refuses when called.
 */
FMI3_Export fmi3Status fmi3ActivateModelPartition(fmi3Instance instance, fmi3ValueReference clockReference, fmi3Float64 activationTime);
FMI3_Export fmi3Status fmi3DeserializeFMUState(fmi3Instance instance, const fmi3Byte serializedState[], size_t size, fmi3FMUState* FMUState);
FMI3_Export fmi3Status fmi3DoStep(fmi3Instance instance, fmi3Float64 currentCommunicationPoint, fmi3Float64 communicationStepSize, fmi3Boolean noSetFMUStatePriorToCurrentPoint, fmi3Boolean* eventHandlingNeeded, fmi3Boolean* terminateSimulation, fmi3Boolean* earlyReturn, fmi3Float64* lastSuccessfulTime);
FMI3_Export fmi3Status fmi3EnterConfigurationMode(fmi3Instance instance);
FMI3_Export fmi3Status fmi3EnterStepMode(fmi3Instance instance);
FMI3_Export fmi3Status fmi3EvaluateDiscreteStates(fmi3Instance instance);
FMI3_Export fmi3Status fmi3ExitConfigurationMode(fmi3Instance instance);
FMI3_Export fmi3Status fmi3FreeFMUState(fmi3Instance instance, fmi3FMUState* FMUState);
FMI3_Export fmi3Status fmi3GetAdjointDerivative(fmi3Instance instance, const fmi3ValueReference unknowns[], size_t nUnknowns, const fmi3ValueReference knowns[], size_t nKnowns, const fmi3Float64 seed[], size_t nSeed, fmi3Float64 sensitivity[], size_t nSensitivity);
FMI3_Export fmi3Status fmi3GetBinary(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, size_t valueSizes[], fmi3Binary values[], size_t nValues);
FMI3_Export fmi3Status fmi3GetBoolean(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, fmi3Boolean values[], size_t nValues);
FMI3_Export fmi3Status fmi3GetClock(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, fmi3Clock values[]);
FMI3_Export fmi3Status fmi3GetFMUState(fmi3Instance instance, fmi3FMUState* FMUState);
FMI3_Export fmi3Status fmi3GetFloat32(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, fmi3Float32 values[], size_t nValues);
FMI3_Export fmi3Status fmi3GetInt16(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, fmi3Int16 values[], size_t nValues);
FMI3_Export fmi3Status fmi3GetInt32(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, fmi3Int32 values[], size_t nValues);
FMI3_Export fmi3Status fmi3GetInt64(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, fmi3Int64 values[], size_t nValues);
FMI3_Export fmi3Status fmi3GetInt8(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, fmi3Int8 values[], size_t nValues);
FMI3_Export fmi3Status fmi3GetIntervalDecimal(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, fmi3Float64 intervals[], fmi3IntervalQualifier qualifiers[]);
FMI3_Export fmi3Status fmi3GetIntervalFraction(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, fmi3UInt64 counters[], fmi3UInt64 resolutions[], fmi3IntervalQualifier qualifiers[]);
FMI3_Export fmi3Status fmi3GetNumberOfVariableDependencies(fmi3Instance instance, fmi3ValueReference valueReference, size_t* nDependencies);
FMI3_Export fmi3Status fmi3GetOutputDerivatives(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, const fmi3Int32 orders[], fmi3Float64 values[], size_t nValues);
FMI3_Export fmi3Status fmi3GetShiftDecimal(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, fmi3Float64 shifts[]);
FMI3_Export fmi3Status fmi3GetShiftFraction(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, fmi3UInt64 counters[], fmi3UInt64 resolutions[]);
FMI3_Export fmi3Status fmi3GetString(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, fmi3String values[], size_t nValues);
FMI3_Export fmi3Status fmi3GetUInt16(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, fmi3UInt16 values[], size_t nValues);
FMI3_Export fmi3Status fmi3GetUInt32(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, fmi3UInt32 values[], size_t nValues);
FMI3_Export fmi3Status fmi3GetUInt64(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, fmi3UInt64 values[], size_t nValues);
FMI3_Export fmi3Status fmi3GetUInt8(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, fmi3UInt8 values[], size_t nValues);
FMI3_Export fmi3Status fmi3GetVariableDependencies(fmi3Instance instance, fmi3ValueReference dependent, size_t elementIndicesOfDependent[], fmi3ValueReference independents[], size_t elementIndicesOfIndependents[], fmi3DependencyKind dependencyKinds[], size_t nDependencies);
FMI3_Export fmi3Instance fmi3InstantiateCoSimulation(fmi3String instanceName, fmi3String instantiationToken, fmi3String resourcePath, fmi3Boolean visible, fmi3Boolean loggingOn, fmi3Boolean eventModeUsed, fmi3Boolean earlyReturnAllowed, const fmi3ValueReference requiredIntermediateVariables[], size_t nRequiredIntermediateVariables, fmi3InstanceEnvironment instanceEnvironment, fmi3LogMessageCallback logMessage, fmi3IntermediateUpdateCallback intermediateUpdate);
FMI3_Export fmi3Instance fmi3InstantiateScheduledExecution(fmi3String instanceName, fmi3String instantiationToken, fmi3String resourcePath, fmi3Boolean visible, fmi3Boolean loggingOn, fmi3InstanceEnvironment instanceEnvironment, fmi3LogMessageCallback logMessage, fmi3ClockUpdateCallback clockUpdate, fmi3LockPreemptionCallback lockPreemption, fmi3UnlockPreemptionCallback unlockPreemption);
FMI3_Export fmi3Status fmi3SerializeFMUState(fmi3Instance instance, fmi3FMUState FMUState, fmi3Byte serializedState[], size_t size);
FMI3_Export fmi3Status fmi3SerializedFMUStateSize(fmi3Instance instance, fmi3FMUState FMUState, size_t* size);
FMI3_Export fmi3Status fmi3SetBinary(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, const size_t valueSizes[], const fmi3Binary values[], size_t nValues);
FMI3_Export fmi3Status fmi3SetBoolean(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, const fmi3Boolean values[], size_t nValues);
FMI3_Export fmi3Status fmi3SetClock(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, const fmi3Clock values[]);
FMI3_Export fmi3Status fmi3SetFMUState(fmi3Instance instance, fmi3FMUState FMUState);
FMI3_Export fmi3Status fmi3SetFloat32(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, const fmi3Float32 values[], size_t nValues);
FMI3_Export fmi3Status fmi3SetInt16(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, const fmi3Int16 values[], size_t nValues);
FMI3_Export fmi3Status fmi3SetInt32(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, const fmi3Int32 values[], size_t nValues);
FMI3_Export fmi3Status fmi3SetInt64(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, const fmi3Int64 values[], size_t nValues);
FMI3_Export fmi3Status fmi3SetInt8(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, const fmi3Int8 values[], size_t nValues);
FMI3_Export fmi3Status fmi3SetIntervalDecimal(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, const fmi3Float64 intervals[]);
FMI3_Export fmi3Status fmi3SetIntervalFraction(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, const fmi3UInt64 counters[], const fmi3UInt64 resolutions[]);
FMI3_Export fmi3Status fmi3SetShiftDecimal(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, const fmi3Float64 shifts[]);
FMI3_Export fmi3Status fmi3SetShiftFraction(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, const fmi3UInt64 counters[], const fmi3UInt64 resolutions[]);
FMI3_Export fmi3Status fmi3SetString(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, const fmi3String values[], size_t nValues);
FMI3_Export fmi3Status fmi3SetUInt16(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, const fmi3UInt16 values[], size_t nValues);
FMI3_Export fmi3Status fmi3SetUInt32(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, const fmi3UInt32 values[], size_t nValues);
FMI3_Export fmi3Status fmi3SetUInt64(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, const fmi3UInt64 values[], size_t nValues);
FMI3_Export fmi3Status fmi3SetUInt8(fmi3Instance instance, const fmi3ValueReference valueReferences[], size_t nValueReferences, const fmi3UInt8 values[], size_t nValues);

#ifdef __cplusplus
}
#endif

#endif /* FASTSIM_FMI3_H */
"#;
