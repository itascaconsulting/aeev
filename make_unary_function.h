#define UNARY_S_FUNC(base_name, func_name)                      \
case S_##base_name: {                                           \
  d_stack[p_d-1] = func_name( d_stack[p_d-1]);\
  break;\
}\
case AS_##base_name: {                                           \
  for (k=0; k<chunk; k++) {\
    res[k] = func_name(a[k]);\
  }\
  break;\
}
