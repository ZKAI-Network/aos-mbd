#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <lua.h>
#include <lauxlib.h>
#include "onnxruntime_c_api.h"

const OrtApi* g_ort = NULL;

#define ORT_LUA_ERROR(L, expr)                                \
do {                                                            \
OrtStatus* onnx_status = (expr);                                \
    if (onnx_status != NULL) {                                  \
        const char* msg = g_ort->GetErrorMessage(onnx_status);  \
        g_ort->ReleaseStatus(onnx_status);                      \
        luaL_error((L), "[ORT] %s\n", msg);                     \
        return 0;                                               \
    }                                                           \
} while (0)

const static char* lort_tensort_elemennt_data_type [] = {
    "UNDEFINED",
    "FLOAT",
    "UINT8",
    "INT8",
    "UINT16",
    "INT16",
    "INT32",
    "INT64",
    "STRING",
    "BOOL",
    "FLOAT16",
    "DOUBLE",
    "UINT32",
    "UINT64",
    NULL
};

static const char* lort_AllocatorType [] = {
    "Invalid",
    "Device",
    "Arena",
    NULL
};

static const char* lort_MemType [] = {
    "CPUInput",
    "CPUOutput",
    "Default",
    NULL
};

static size_t getsize(ONNXTensorElementDataType datatype) {
    switch (datatype)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return (sizeof(float)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return (sizeof(uint8_t)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return (sizeof(int8_t)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return (sizeof(uint16_t)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return (sizeof(int16_t)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return (sizeof(int32_t)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return (sizeof(int64_t)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return (sizeof(int16_t)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return (sizeof(double)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return (sizeof(uint32_t)); break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return (sizeof(uint64_t)); break;
    default:
        break;
    }
    return 0;
}

static int64_t * lort_toiarray(lua_State *L, int index, size_t* count, int64_t* elements_count) {
    *count = lua_rawlen(L, index);
    int64_t *array = calloc(*count, sizeof(int64_t));
    *elements_count = 1;
    for (int i = 1; i <= *count; i++) {
        lua_rawgeti(L, index, i);
        array[i-1] = (int64_t)lua_tointeger(L, -1);
        *elements_count *= array[i-1];
        lua_pop(L, 1);
    }

    return array;
}

static int lort_createenv (lua_State *L) {
    OrtEnv* env;
    ORT_LUA_ERROR(L, g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "OrtLua", &env));
    if (env == NULL) { luaL_error(L, "Failed env creating."); }

    OrtEnv** luaptr = (OrtEnv**)lua_newuserdata(L, sizeof(OrtEnv*));
    *luaptr = env;

    luaL_getmetatable(L, "Ort.Env");    
    lua_setmetatable(L, -2);

    return 1;
}

static int lort_createsessionoptions (lua_State *L) {
    OrtSessionOptions* session_options;
    ORT_LUA_ERROR(L, g_ort->CreateSessionOptions(&session_options));
    if (session_options == NULL) { luaL_error(L, "Failed options creating."); }

    OrtSessionOptions** luaptr = (OrtSessionOptions**)lua_newuserdata(L, sizeof(OrtSessionOptions*));
    *luaptr = session_options;

    luaL_getmetatable(L, "Ort.SessionOptions");    
    lua_setmetatable(L, -2);

    return 1;
}

static int lort_createvalue (lua_State *L) {
    luaL_checktype(L, 1, LUA_TTABLE);
    luaL_checktype(L, 3, LUA_TTABLE);
    int element_data_type = luaL_checkoption(L, 2, "FLOAT", lort_tensort_elemennt_data_type);

    size_t input_shape_len;
    int64_t elements_count;
    int64_t *input_shape = lort_toiarray(L, 1, &input_shape_len, &elements_count);

    OrtAllocator* default_allocator = NULL;
    g_ort->GetAllocatorWithDefaultOptions(&default_allocator);

    OrtValue* input_tensor = NULL;
    ORT_LUA_ERROR(L, g_ort->CreateTensorAsOrtValue(default_allocator, input_shape, input_shape_len, element_data_type, &input_tensor));
    luaL_argcheck(L, input_tensor != NULL, 1, "Failed creating tensor");

    if (lua_istable(L, 3)) {
        if (element_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
            int strings_count = (int)luaL_len(L, 3);

            const char** s = calloc(sizeof(char* const*), strings_count);

            for (int i = 1; i <= strings_count; i++) {
                lua_geti(L, 3, i);
                s[i] = lua_tostring(L, -1);
                lua_pop(L, 1);
            }

            ORT_LUA_ERROR(L, g_ort->FillStringTensor(input_tensor, s, strings_count));

            free(s);
        } else {
            lua_Integer modelort_input_ele_count = luaL_len(L, 3);
            char* modelort_inputc = NULL;
            ORT_LUA_ERROR(L, g_ort->GetTensorMutableData(input_tensor, (void **)&modelort_inputc));
            lua_Integer count = (lua_Integer)(modelort_input_ele_count < elements_count ? modelort_input_ele_count : elements_count);

            for (lua_Integer i = 0; i < count; i++) {
                lua_geti(L, 3, i + 1);
                if (lua_isnumber(L, -1) == 0) {
                    g_ort->ReleaseValue(input_tensor);
                    luaL_error(L, "Data must be number table");
                    return 0;
                }
                
                lua_Number el = lua_tonumber(L, -1);
                switch (element_data_type)
                {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                    ((float*)modelort_inputc)[i] = (float) el;
                    break;

                case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                    ((double*)modelort_inputc)[i] = (double) el;
                    break;
                
                default:
                    luaL_error(L, "Not implemented tensor type %i", (int)element_data_type);
                    break;
                }
                lua_pop(L, 1);
            }
        }
    }

    OrtValue** luaptr = (OrtValue**)lua_newuserdata(L, sizeof(OrtValue*));
    *luaptr = input_tensor;
    luaL_getmetatable(L, "Ort.Value");    
    lua_setmetatable(L, -2);

    free(input_shape);

    return 1;
}

static const struct luaL_Reg ort [] = {
    {"CreateEnv", lort_createenv},
    {"CreateSessionOptions", lort_createsessionoptions},
    {"CreateValue", lort_createvalue},
    {NULL, NULL}
};


// Env

static int lort_env_createsession (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    luaL_checktype(L, 2, LUA_TSTRING);
    luaL_checktype(L, 3, LUA_TUSERDATA);

    OrtEnv* env = *(OrtEnv**)luaL_checkudata(L, 1, "Ort.Env");
    size_t pathlen;
    const char* modelpath = lua_tolstring(L, 2, &pathlen);
    wchar_t* wmodelpath = (wchar_t*)calloc(pathlen+1, sizeof(wchar_t));
    mbstowcs(wmodelpath, modelpath, pathlen);
    OrtSessionOptions* session_options = *(OrtSessionOptions**)luaL_checkudata(L, 3, "Ort.SessionOptions");

    OrtSession* session;
    ORT_LUA_ERROR(L, g_ort->CreateSession(env, (const char *)wmodelpath, session_options, &session));
    if (session == NULL) { luaL_error(L, "Failed env creating."); }

    OrtSession** luaptr = (OrtSession**)lua_newuserdata(L, sizeof(OrtSession*));
    *luaptr = session;
    luaL_getmetatable(L, "Ort.Session");    
    lua_setmetatable(L, -2);

    free(wmodelpath);

    return 1;
}

static int lort_env_release (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtEnv* env = *(OrtEnv**)luaL_checkudata(L, 1, "Ort.Env");

    g_ort->ReleaseEnv(env);

    return 0;
}

static const struct luaL_Reg env_m [] = {
    {"CreateSession", lort_env_createsession},
    {"__gc", lort_env_release},
    {NULL, NULL}
};


// SessionOptions

static int lort_sessionoptions_AppendExecutionProvider_DML (lua_State *L) {
    #ifdef USE_DML
        luaL_checktype(L, 1, LUA_TUSERDATA);
        OrtSessionOptions* session_options = *(OrtSessionOptions**)luaL_checkudata(L, 1, "Ort.SessionOptions");
        ORT_LUA_ERROR(L, g_ort->OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
    #else
        luaL_error(L, "DirectML is not enabled in this build.");
    #endif
    
    return 0;
}

static int lort_sessionoptions_AppendExecutionProvider_OpenVINO (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtSessionOptions* session_options = *(OrtSessionOptions**)luaL_checkudata(L, 1, "Ort.SessionOptions");

    OrtOpenVINOProviderOptions provider_options;
    memset(&provider_options, 0, sizeof(provider_options));

    if (lua_istable(L, 2)) {
        lua_getfield(L, 2, "device_type");
        provider_options.device_type = lua_tostring(L, -1);
    }

    ORT_LUA_ERROR(L, g_ort->SessionOptionsAppendExecutionProvider_OpenVINO(session_options, &provider_options));

    return 0;
}

static int lort_sessionoptions_AppendExecutionProvider_CUDA (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtSessionOptions* session_options = *(OrtSessionOptions**)luaL_checkudata(L, 1, "Ort.SessionOptions");

    OrtCUDAProviderOptions o;
    // Here we use memset to initialize every field of the above data struct to zero.
    memset(&o, 0, sizeof(o));

    o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    o.gpu_mem_limit = SIZE_MAX;
    ORT_LUA_ERROR(L, g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options, &o));
    
    return 0;
}

static int lort_sessionoptions_AppendExecutionProvider (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    luaL_checktype(L, 3, LUA_TTABLE);
    OrtSessionOptions* session_options = *(OrtSessionOptions**)luaL_checkudata(L, 1, "Ort.SessionOptions");
    const char* provider_name = luaL_checkstring(L, 2);
    size_t options = 16;
    const char** keys = calloc(options, sizeof(char*));
    const char** values = calloc(options, sizeof(char*));

    int index = 0;
    lua_pushnil(L);
    while(lua_next(L, 3) != 0) {
        if (index > options) {
            options = options * 2;
            keys = realloc(keys, options * sizeof(char*));
            values = realloc(values, options * sizeof(char*));
        }
        keys[index] = lua_tostring(L, -2);
        values[index] = lua_tostring(L, -1);
        index++;
        lua_pop(L, 1);
    }
    ORT_LUA_ERROR(L, g_ort->SessionOptionsAppendExecutionProvider(session_options, provider_name, keys, values, index));

    free(keys);
    free(values);
    return 0;
}

static int lort_sessionoptions_release (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtSessionOptions* session_options = *(OrtSessionOptions**)luaL_checkudata(L, 1, "Ort.SessionOptions");

    g_ort->ReleaseSessionOptions(session_options);

    return 0;
}

static const struct luaL_Reg sessionoptions_m [] = {
    {"AppendExecutionProvider_DML", lort_sessionoptions_AppendExecutionProvider_DML},
    {"AppendExecutionProvider_CUDA", lort_sessionoptions_AppendExecutionProvider_CUDA},
    {"AppendExecutionProvider_OpenVINO", lort_sessionoptions_AppendExecutionProvider_OpenVINO},
    {"AppendExecutionProvider", lort_sessionoptions_AppendExecutionProvider},
    {"__gc", lort_sessionoptions_release},
    {NULL, NULL}
};


// Session

static int lort_session_GetInputs(lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtSession* session = *(OrtSession**)luaL_checkudata(L, 1, "Ort.Session");
    
    OrtAllocator* allocator = NULL;
    ORT_LUA_ERROR(L, g_ort->GetAllocatorWithDefaultOptions(&allocator));

    size_t count;
    ORT_LUA_ERROR(L, g_ort->SessionGetInputCount(session, &count));

    lua_createtable(L, (int)count, 0);

    char* value;
    for (size_t i = 0; i < count; i++) {
        ORT_LUA_ERROR(L, g_ort->SessionGetInputName(session, i, allocator, &value));

        lua_pushstring(L, value);
        lua_rawseti(L, -2, (lua_Integer)i + 1);

        allocator->Free(allocator, value);
    }

    return 1;
}

static int lort_session_GetOutputs(lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtSession* session = *(OrtSession**)luaL_checkudata(L, 1, "Ort.Session");
    
    OrtAllocator* allocator = NULL;
    ORT_LUA_ERROR(L, g_ort->GetAllocatorWithDefaultOptions(&allocator));

    size_t count;
    ORT_LUA_ERROR(L, g_ort->SessionGetOutputCount(session, &count));

    lua_createtable(L, (int)count, 0);

    char* value;
    for (size_t i = 0; i < count; i++) {
        ORT_LUA_ERROR(L, g_ort->SessionGetOutputName(session, i, allocator, &value));

        lua_pushstring(L, value);
        lua_rawseti(L, -2, (lua_Integer)i + 1);

        allocator->Free(allocator, value);
    }

    return 1;
}

static int lort_session_GetInputType(lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    int index = (int)luaL_checkinteger(L, 2);
    OrtSession* session = *(OrtSession**)luaL_checkudata(L, 1, "Ort.Session");

    OrtTypeInfo* type_info = NULL;
    ORT_LUA_ERROR(L, g_ort->SessionGetInputTypeInfo(session, index - 1, &type_info));

    OrtTensorTypeAndShapeInfo *tensor_info = NULL;
    ORT_LUA_ERROR(L, g_ort->CastTypeInfoToTensorInfo(type_info, (const OrtTensorTypeAndShapeInfo**)&tensor_info));

    ONNXTensorElementDataType type;
    ORT_LUA_ERROR(L, g_ort->GetTensorElementType(tensor_info, &type));

    lua_pushstring(L, lort_tensort_elemennt_data_type[type]);

    size_t dims_count;
    ORT_LUA_ERROR(L, g_ort->GetDimensionsCount(tensor_info, &dims_count));

    int64_t* dims = calloc(dims_count, sizeof(int64_t));
    ORT_LUA_ERROR(L, g_ort->GetDimensions(tensor_info, dims, dims_count));

    lua_createtable(L, (int)dims_count, 0);
    for (size_t i = 0; i < dims_count; i++) {
        lua_pushinteger(L, dims[i]);
        lua_rawseti(L, -2, (lua_Integer)i + 1);
    }

    free(dims);
    g_ort->ReleaseTypeInfo(type_info);
    return 2;
}

static int lort_session_GetOutputType(lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    int index = (int)luaL_checkinteger(L, 2);
    OrtSession* session = *(OrtSession**)luaL_checkudata(L, 1, "Ort.Session");

    OrtTypeInfo* type_info = NULL;
    ORT_LUA_ERROR(L, g_ort->SessionGetOutputTypeInfo(session, index - 1, &type_info));

    OrtTensorTypeAndShapeInfo *tensor_info = NULL;
    ORT_LUA_ERROR(L, g_ort->CastTypeInfoToTensorInfo(type_info, (const OrtTensorTypeAndShapeInfo**)&tensor_info));

    ONNXTensorElementDataType type;
    ORT_LUA_ERROR(L, g_ort->GetTensorElementType(tensor_info, &type));

    lua_pushstring(L, lort_tensort_elemennt_data_type[type]);

    size_t dims_count;
    ORT_LUA_ERROR(L, g_ort->GetDimensionsCount(tensor_info, &dims_count));

    int64_t* dims = calloc(dims_count, sizeof(int64_t));
    ORT_LUA_ERROR(L, g_ort->GetDimensions(tensor_info, dims, dims_count));

    lua_createtable(L, (int)dims_count, 0);
    for (size_t i = 0; i < dims_count; i++) {
        lua_pushinteger(L, dims[i]);
        lua_rawseti(L, -2, (lua_Integer)i + 1);
    }

    free(dims);
    g_ort->ReleaseTypeInfo(type_info);
    return 2;
}

static int lort_session_release(lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtSession* session = *(OrtSession**)luaL_checkudata(L, 1, "Ort.Session");

    g_ort->ReleaseSession(session);

    return 0;
}

static int lort_session_run (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    luaL_checktype(L, 2, LUA_TTABLE);
    OrtSession* session = *(OrtSession**)luaL_checkudata(L, 1, "Ort.Session");
    OrtAllocator* allocator = NULL;
    size_t input_count;
    size_t output_count;

    ORT_LUA_ERROR(L, g_ort->GetAllocatorWithDefaultOptions(&allocator));
    ORT_LUA_ERROR(L, g_ort->SessionGetInputCount(session, &input_count));
    ORT_LUA_ERROR(L, g_ort->SessionGetOutputCount(session, &output_count));

    const char** input_names = calloc(input_count, sizeof(char*));
    const char** output_names = calloc(output_count, sizeof(char*));
    const OrtValue** input_tensors = calloc(input_count, sizeof(OrtValue*));
    OrtValue** output_tensors = calloc(output_count, sizeof(OrtValue*));
    
    int index = 0;
    lua_pushnil(L);
    while (lua_next(L, 2) != 0) {
        if (index > input_count) {
            free(input_names);
            free(output_names);
            free(input_tensors);
            free(output_tensors);
            luaL_error(L, "Input count mistmatch");
            return 0;
        }
        input_tensors[index] = *(OrtValue**)luaL_checkudata(L, -1, "Ort.Value");
        input_names[index] = lua_tostring(L, -2);
        index++;
        lua_pop(L, 1);
    }
    lua_pop(L, 1);

    for (size_t i = 0; i < output_count; i++) {
        ORT_LUA_ERROR(L, g_ort->SessionGetOutputName(session, i, allocator, (char **)output_names + i));
    }

    ORT_LUA_ERROR(L, g_ort->Run(session, NULL, input_names, 
                                    input_tensors,
                                    index, output_names,
                                    output_count, output_tensors));
    luaL_argcheck(L, output_tensors != NULL, 1, "Failed runing");

    lua_createtable(L, (int)output_count, 0);
    for (int i = 0; i < output_count; i++) {
        const OrtValue** luaptr = (const OrtValue **)lua_newuserdata(L, sizeof(OrtValue*));
        *luaptr = output_tensors[i];
        luaL_getmetatable(L, "Ort.Value");    
        lua_setmetatable(L, -2);
        lua_setfield(L, -2, output_names[i]);
        allocator->Free(allocator, (void *)output_names[i]);
    }

    free(input_names);
    free(output_names);
    free(input_tensors);
    free(output_tensors);
    return 1;
}

static const struct luaL_Reg session_m [] = {
    {"GetInputs", lort_session_GetInputs},
    {"GetOutputs", lort_session_GetOutputs},
    {"GetInputType", lort_session_GetInputType},
    {"GetOutputType", lort_session_GetOutputType},
    {"Run", lort_session_run},
    {"__gc", lort_session_release},
    {NULL, NULL}
};

// OrtValue

static int lort_value_istensor (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtValue* value = *(OrtValue**)luaL_checkudata(L, 1, "Ort.Value");

    int is_tensor;
    g_ort->IsTensor(value, &is_tensor);

    lua_pushboolean(L, is_tensor);

    return 1;
}

static int lort_value_getdata (lua_State *L) { // TODO improve data size
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtValue* value = *(OrtValue**)luaL_checkudata(L, 1, "Ort.Value");

    OrtTensorTypeAndShapeInfo* typeandshape = NULL;
    ORT_LUA_ERROR(L, g_ort->GetTensorTypeAndShape(value, &typeandshape));

    size_t count;
    ORT_LUA_ERROR(L, g_ort->GetTensorShapeElementCount(typeandshape, &count));
    ONNXTensorElementDataType datatype;
    ORT_LUA_ERROR(L, g_ort->GetTensorElementType(typeandshape, &datatype));

    //size_t sizeofel = getsize(datatype);

    char* output_tensor_data = NULL;
    ORT_LUA_ERROR(L, g_ort->GetTensorMutableData(value, (void**)&output_tensor_data));

    lua_createtable(L, count, 2); // reserve for fields for recording in PNG

    switch (datatype)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        for (size_t i = 0; i < count; ++i) {
            float d = ((float*)output_tensor_data)[i];
            lua_pushnumber(L, d);
            lua_rawseti(L, -2, (lua_Integer)(i + 1));
        }
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        for (size_t i = 0; i < count; ++i) {
            lua_pushnumber(L, ((double*)output_tensor_data)[i]);
            lua_rawseti(L, -2, (lua_Integer)(i + 1));
        }
        break;
    
    default:
        luaL_error(L, "Not implemented tensor type %i", (int)datatype);
        break;
    }

    g_ort->ReleaseTensorTypeAndShapeInfo(typeandshape);

    return 1;
}

static int lort_value_release (lua_State *L) {
    luaL_checktype(L, 1, LUA_TUSERDATA);
    OrtValue* value = *(OrtValue**)luaL_checkudata(L, 1, "Ort.Value");

    g_ort->ReleaseValue(value);

    return 0;
}

static const struct luaL_Reg value_m [] = {
    {"isTensor", lort_value_istensor},
    {"GetData", lort_value_getdata},
    {"__gc", lort_value_release},
    {NULL, NULL}
};

/* Remaining classes
ORT_RUNTIME_CLASS(IoBinding);
ORT_RUNTIME_CLASS(RunOptions);
ORT_RUNTIME_CLASS(TypeInfo);
ORT_RUNTIME_CLASS(TensorTypeAndShapeInfo);
ORT_RUNTIME_CLASS(CustomOpDomain);
ORT_RUNTIME_CLASS(MapTypeInfo);
ORT_RUNTIME_CLASS(SequenceTypeInfo);
ORT_RUNTIME_CLASS(ModelMetadata);
ORT_RUNTIME_CLASS(ThreadPoolParams);
ORT_RUNTIME_CLASS(ThreadingOptions);
ORT_RUNTIME_CLASS(ArenaCfg);
ORT_RUNTIME_CLASS(PrepackedWeightsContainer);
ORT_RUNTIME_CLASS(TensorRTProviderOptionsV2);
*/

#if defined(_WIN32) || defined(_WIN64)
__declspec(dllexport)
#endif
int luaopen_ort(lua_State *L) {

    luaL_newmetatable(L, "Ort.Env");
    lua_pushvalue(L, -1);
    lua_setfield(L, -2, "__index");
    luaL_setfuncs(L, env_m, 0);

    luaL_newmetatable(L, "Ort.Session");
    lua_pushvalue(L, -1);
    lua_setfield(L, -2, "__index");
    luaL_setfuncs(L, session_m, 0);

    luaL_newmetatable(L, "Ort.SessionOptions");
    lua_pushvalue(L, -1);
    lua_setfield(L, -2, "__index");
    luaL_setfuncs(L, sessionoptions_m, 0);

    luaL_newmetatable(L, "Ort.Value");
    lua_pushvalue(L, -1);
    lua_setfield(L, -2, "__index");
    luaL_setfuncs(L, value_m, 0);

    luaL_newlib(L, ort);

    return 1;
}