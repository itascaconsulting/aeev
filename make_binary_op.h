#define BINARY_OPERATOR(base_name, oper)\
\
case S_S_##base_name: {\
  d_stack[p_d-2] = d_stack[p_d-2] oper d_stack[p_d-1];\
  p_d--;\
  break;\
}\
case S_AS_##base_name: {\
  for (k=0; k<chunk; k++) {\
    res[k] = d_stack[p_d-1] oper b[k];\
  }\
  p_d--;\
  break;\
}\
case S_V_##base_name: {\
  double tmp = d_stack[p_d-4];\
  d_stack[p_d-4] = tmp oper d_stack[p_d-3];\
  d_stack[p_d-3] = tmp oper d_stack[p_d-2];\
  d_stack[p_d-2] = tmp oper d_stack[p_d-1];\
  p_d--;\
  break;\
}\
case S_AV_##base_name: {\
  for (k=0; k<chunk; k++) {\
    res[k*3  ] = d_stack[p_d-1] oper b[k*3  ];\
    res[k*3+1] = d_stack[p_d-1] oper b[k*3+1];\
    res[k*3+2] = d_stack[p_d-1] oper b[k*3+2];\
  }\
  p_d--;\
  break;\
}\
\
case AS_S_##base_name: {\
  for (k=0; k<chunk; k++) {\
    res[k] = a[k] oper d_stack[p_d-1];\
  }\
  p_d--;\
  break;\
}\
case AS_AS_##base_name: {\
  for (k=0; k<chunk; k++) {\
    res[k] = a[k] oper b[k];\
  }\
  break;\
}\
case AS_V_##base_name: {\
  for (k=0; k<chunk; k++) {\
    res[k*3  ] = a[k] oper d_stack[p_d-3];\
    res[k*3+1] = a[k] oper d_stack[p_d-2];\
    res[k*3+2] = a[k] oper d_stack[p_d-1];\
  }\
  p_d -= 3;\
  break;\
}\
case AS_AV_##base_name: {\
  for (k=0; k<chunk; k++) {\
    res[k*3  ] = a[k] oper b[3*k  ];\
    res[k*3+1] = a[k] oper b[3*k+1];\
    res[k*3+2] = a[k] oper b[3*k+2];\
  }\
  break;\
}\
\
case V_S_##base_name: {\
  d_stack[p_d-4] =  d_stack[p_d-4] oper d_stack[p_d-1];\
  d_stack[p_d-3] =  d_stack[p_d-3] oper d_stack[p_d-1];\
  d_stack[p_d-2] =  d_stack[p_d-2] oper d_stack[p_d-1];\
  p_d--;\
  break;\
}\
case V_AS_##base_name: {\
  for (k=0; k<chunk; k++) {\
    res[k*3  ] = d_stack[p_d-3] oper b[k];\
    res[k*3+1] = d_stack[p_d-2] oper b[k];\
    res[k*3+2] = d_stack[p_d-1] oper b[k];\
  }\
  p_d -= 3;\
  break;\
}\
case V_V_##base_name: {\
  d_stack[p_d-6] =  d_stack[p_d-6] oper d_stack[p_d-3];\
  d_stack[p_d-5] =  d_stack[p_d-5] oper d_stack[p_d-2];\
  d_stack[p_d-4] =  d_stack[p_d-4] oper d_stack[p_d-1];\
  p_d -= 3;\
  break;\
}\
case V_AV_##base_name: {\
  for (k=0; k<chunk; k++) {\
    res[k*3  ] = d_stack[p_d-3] oper b[k*3  ];\
    res[k*3+1] = d_stack[p_d-2] oper b[k*3+1];\
    res[k*3+2] = d_stack[p_d-1] oper b[k*3+2];\
  }\
  p_d -= 3;\
  break;\
}\
\
case AV_S_##base_name: {\
  for (k=0; k<chunk; k++) {\
    res[k*3  ] = a[k*3  ] oper d_stack[p_d-1];\
    res[k*3+1] = a[k*3+1] oper d_stack[p_d-1];\
    res[k*3+2] = a[k*3+2] oper d_stack[p_d-1];\
  }\
  p_d--;\
  break;\
}\
case AV_AS_##base_name: {\
  for (k=0; k<chunk; k++) {\
    res[k*3  ] = a[k*3  ] oper b[k];\
    res[k*3+1] = a[k*3+1] oper b[k];\
    res[k*3+2] = a[k*3+2] oper b[k];\
  }\
  break;\
}\
case AV_V_##base_name: {\
  for (k=0; k<chunk; k++) {\
    res[k*3  ] = a[k*3  ] oper d_stack[p_d-3];\
    res[k*3+1] = a[k*3+1] oper d_stack[p_d-2];\
    res[k*3+2] = a[k*3+2] oper d_stack[p_d-1];\
  }\
  p_d -= 3;\
  break;\
}\
case AV_AV_##base_name: {\
  for (k=0; k<chunk; k++) {\
    res[k*3  ] = a[k*3  ] oper b[k*3  ];\
    res[k*3+1] = a[k*3+1] oper b[k*3+1];\
    res[k*3+2] = a[k*3+2] oper b[k*3+2];\
  }\
  break;\
}
