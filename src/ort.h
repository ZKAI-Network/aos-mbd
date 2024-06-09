#ifndef ORT_H
#define ORT_H
#include <stdbool.h>
#include "onnxruntime_c_api.h"
#include "lua.h"

int luaopen_ort(lua_State *L);

// size_t getsize(ONNXTensorElementDataType datatype);
// int64_t * lort_toiarray(lua_State *L, int index, size_t* count, int64_t* elements_count);
int lort_createenv (lua_State *L);
// int lort_createsessionoptions (lua_State *L);
// int lort_createvalue (lua_State *L);

// // Env

// int lort_env_createsession (lua_State *L);
// int lort_env_release (lua_State *L);

// // SessionOptions

// int lort_sessionoptions_AppendExecutionProvider_DML (lua_State *L);
// int lort_sessionoptions_AppendExecutionProvider_OpenVINO (lua_State *L);
// int lort_sessionoptions_AppendExecutionProvider_CUDA (lua_State *L);
// int lort_sessionoptions_AppendExecutionProvider (lua_State *L);
// int lort_sessionoptions_release (lua_State *L);

// // Session

// int lort_session_GetInputs(lua_State *L);
// int lort_session_GetOutputs(lua_State *L);
// int lort_session_GetInputType(lua_State *L);
// int lort_session_GetOutputType(lua_State *L);
// int lort_session_release(lua_State *L);
// int lort_session_run (lua_State *L);

// // OrtValue

// int lort_value_istensor (lua_State *L);
// int lort_value_getdata (lua_State *L);
// int lort_value_release (lua_State *L);
#endif