#define OPERATOR(base_name, oper)           \
case AS_AS_##base_name:\
for (k=0; k<chunk; k++) {res[k] = a[k] oper b[k];} break;\
case AS_S_##base_name:\
for (k=0; k<chunk; k++) { res[k] = a[k] oper dstack[dstack_ptr-1];}\
dstack_ptr--; break;\
case S_AS_##base_name:\
for (k=0; k<chunk; k++) { res[k] = dstack[dstack_ptr-1] oper b[k];}\
dstack_ptr--; break;\
case S_S_##base_name:\
dstack[dstack_ptr-2] = dstack[dstack_ptr-2] oper dstack[dstack_ptr-1];\
dstack_ptr--; break;
