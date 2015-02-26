#define OPERATOR(base_name, oper)           \
case A_A_##base_name:\
for (k=0; k<CHUNK_SIZE; k++) {res[k] = a[k] oper b[k];} break;\
case A_S_##base_name:\
for (k=0; k<CHUNK_SIZE; k++) { res[k] = a[k] oper dstack[dstack_ptr-1];}\
dstack_ptr--; break;\
case S_A_##base_name:\
for (k=0; k<CHUNK_SIZE; k++) { res[k] = dstack[dstack_ptr-1] oper b[k];}\
dstack_ptr--; break;\
case S_S_##base_name:\
dstack[dstack_ptr-2] = dstack[dstack_ptr-2] oper dstack[dstack_ptr-1];\
dstack_ptr--; break;
