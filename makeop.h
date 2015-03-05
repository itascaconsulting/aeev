#define OPERATOR(base_name, oper)\
\
case S_S_##base_name: {\
  dstack[dstack_ptr-2] = dstack[dstack_ptr-2] oper dstack[dstack_ptr-1];\
  dstack_ptr--;\
  break;\
}\
case S_AS_##base_name: {\
  for (k=0; k<chunk; k++) {\
    res[k] = dstack[dstack_ptr-1] oper b[k];\
  }\
  dstack_ptr--;\
  break;\
}\
case S_V_##base_name: {\
  double tmp = dstack[dstack_ptr-4];\
  dstack[dstack_ptr-4] = tmp oper dstack[dstack_ptr-3];\
  dstack[dstack_ptr-3] = tmp oper dstack[dstack_ptr-2];\
  dstack[dstack_ptr-2] = tmp oper dstack[dstack_ptr-1];\
  dstack_ptr--;\
  break;\
}\
case S_AV_##base_name: {\
  for (k=0; k<chunk; k++) {\
    res[k*3  ] = dstack[dstack_ptr-1] oper b[k*3  ];\
    res[k*3+1] = dstack[dstack_ptr-1] oper b[k*3+1];\
    res[k*3+2] = dstack[dstack_ptr-1] oper b[k*3+2];\
  }\
  dstack_ptr--;\
  break;\
}\
\
case AS_S_##base_name: {\
  for (k=0; k<chunk; k++) {\
    res[k] = a[k] oper dstack[dstack_ptr-1];\
  }\
  dstack_ptr--;\
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
    res[k*3  ] = a[k] oper dstack[dstack_ptr-3];\
    res[k*3+1] = a[k] oper dstack[dstack_ptr-2];\
    res[k*3+2] = a[k] oper dstack[dstack_ptr-1];\
  }\
  dstack_ptr -= 3;\
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
  dstack[dstack_ptr-4] =  dstack[dstack_ptr-4] oper dstack[dstack_ptr-1];\
  dstack[dstack_ptr-3] =  dstack[dstack_ptr-3] oper dstack[dstack_ptr-1];\
  dstack[dstack_ptr-2] =  dstack[dstack_ptr-2] oper dstack[dstack_ptr-1];\
  dstack_ptr--;\
  break;\
}\
case V_AS_##base_name: {\
  for (k=0; k<chunk; k++) {\
    res[k*3  ] = dstack[dstack_ptr-3] oper b[k];\
    res[k*3+1] = dstack[dstack_ptr-2] oper b[k];\
    res[k*3+2] = dstack[dstack_ptr-1] oper b[k];\
  }\
  dstack_ptr -= 3;\
  break;\
}\
case V_V_##base_name: {\
  dstack[dstack_ptr-6] =  dstack[dstack_ptr-6] oper dstack[dstack_ptr-3];\
  dstack[dstack_ptr-5] =  dstack[dstack_ptr-5] oper dstack[dstack_ptr-2];\
  dstack[dstack_ptr-4] =  dstack[dstack_ptr-4] oper dstack[dstack_ptr-1];\
  dstack_ptr -= 3;\
  break;\
}\
case V_AV_##base_name: {\
  for (k=0; k<chunk; k++) {\
    res[k*3  ] = dstack[dstack_ptr-3] oper b[k*3  ];\
    res[k*3+1] = dstack[dstack_ptr-2] oper b[k*3+1];\
    res[k*3+2] = dstack[dstack_ptr-1] oper b[k*3+2];\
  }\
  dstack_ptr -= 3;\
  break;\
}\
\
case AV_S_##base_name: {\
  for (k=0; k<chunk; k++) {\
    res[k*3  ] = a[k*3  ] oper dstack[dstack_ptr-1];\
    res[k*3+1] = a[k*3+1] oper dstack[dstack_ptr-1];\
    res[k*3+2] = a[k*3+2] oper dstack[dstack_ptr-1];\
  }\
  dstack_ptr--;\
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
    res[k*3  ] = a[k*3  ] oper dstack[dstack_ptr-3];\
    res[k*3+1] = a[k*3+1] oper dstack[dstack_ptr-2];\
    res[k*3+2] = a[k*3+2] oper dstack[dstack_ptr-1];\
  }\
  dstack_ptr -= 3;\
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
