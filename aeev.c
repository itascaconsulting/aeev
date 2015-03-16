#include "Python.h"
#include "numpy/noprefix.h"
#include "stdio.h"
#include "ops.h"

#define CHUNK_SIZE 256
#define GET_HEAP_PTR(arg) (double *) PyArray_DATA((PyArrayObject *) PyTuple_GET_ITEM(array_literals, al_stack[arg]))
#define INVALID PyErr_SetString(PyExc_ValueError, "invalid bytecode"); return 0;
#define STACK_DEPTH 8
#include "make_binary_op.h"
#include "make_unary_function.h"

int process_chunk(int i, int chunk, int nops, double *c_double_literals,
                  double *c_target, long *c_opcodes, PyObject *array_literals) {
    int    al_stack[STACK_DEPTH];
    double as_stack[STACK_DEPTH][CHUNK_SIZE];
    double av_stack[STACK_DEPTH][CHUNK_SIZE*3];
    double d_stack[STACK_DEPTH];
    int p_d=0;  // stack pointers
    int p_as=0;
    int p_av=0;
    int p_al=0;
    int j=0;
    int k=0;
    for (j=0; j<nops; j++) {
        long op = c_opcodes[j];
        if ((op &~ OP_MASK) == LIT_S) {
            d_stack[p_d] = c_double_literals[op & ~BYTECODE_MASK];
            p_d++;
        }
        else if ((op &~ OP_MASK) == LIT_V) {
            d_stack[p_d] = c_double_literals[op & ~BYTECODE_MASK];
            p_d++;
            d_stack[p_d] = c_double_literals[(op & ~BYTECODE_MASK)+1];
            p_d++;
            d_stack[p_d] = c_double_literals[(op & ~BYTECODE_MASK)+2];
            p_d++;
        }
        else if (((op &~ OP_MASK) == LIT_AS) ||
                 ((op &~ OP_MASK) == LIT_AV)) {
            al_stack[p_al] = op & ~BYTECODE_MASK;
            p_al++;
        }
        else  // normal op
        {
            double *res=0; // array result
            double *a = 0; // left
            double *b = 0; // right

            if (op & B_ON_HEAP) {
                if ((op & B_AV) == B_AV) { // b is an av
                    b = GET_HEAP_PTR(p_al-1) + 3*i * CHUNK_SIZE;
                    p_al--;
                } else {  // b is an as
                    b = GET_HEAP_PTR(p_al-1) + i * CHUNK_SIZE;
                    p_al--;
                }
            } else { // b on stack
                if ((op & B_AV) == B_AV) { // b is an av
                    b = av_stack[p_av-1];
                    p_av--;
                } else {
                    if ((op & B_AS) == B_AS)
                    {
                        b = as_stack[p_as-1];
                        p_as--;
                    }
                }
            }
            if (op & A_ON_HEAP) {
                if ((op & A_AV) == A_AV) { // a is an av
                    a = GET_HEAP_PTR(p_al-1) + 3*i * CHUNK_SIZE;
                    p_al--;
                } else {  // b is an as
                    a = GET_HEAP_PTR(p_al-1) + i * CHUNK_SIZE;
                    p_al--;
                }
            } else { // a on stack
                if ((op & A_AV) == A_AV) { // b is an av
                    a = av_stack[p_av-1];
                    p_av--;
                } else {
                    if ((op & A_AS) == A_AS)
                    {
                        a = as_stack[p_as-1];
                        p_as--;
                    }
                }
            }
            if (op & RESULT_TO_HEAP) {
                if ((op & R_AV) == R_AV) {
                    res = c_target + 3 * i * CHUNK_SIZE;
                } else {
                    res = c_target + i * CHUNK_SIZE;
                }
            } else {
                if ((op & R_AV) == R_AV) {
                    res = av_stack[p_av];
                    p_av++;
                } else {
                    if (((op & R_AS) == R_AS)) {
                        res = as_stack[p_as];
                        p_as++;
                    }
                }
            }

            if (p_al >= STACK_DEPTH || p_al < 0 ||
                p_av >= STACK_DEPTH || p_av < 0 ||
                p_as >= STACK_DEPTH || p_as < 0) {
                printf("stack pointers: %i %i %i\n", p_al, p_av, p_as);
                PyErr_SetString(PyExc_ValueError, "Stack overflow.");
                return 0;
            }

            //printf("opcode %i %i %i\n", op, op &~BYTECODE_MASK, case_code);
            switch (op & ~HEAP_MASK) {
                BINARY_OPERATOR(ADD, +);
                BINARY_OPERATOR(SUB, -);
                BINARY_OPERATOR(MUL, *);
                BINARY_OPERATOR(DIV, /);
                UNARY_S_FUNC(EXP, exp);

            case AV_MAG:
                for (k=0; k<chunk; k++) {
                    res[k] = sqrt(a[k*3]*a[k*3] +
                                  a[k*3+1]*a[k*3+1] +
                                  a[k*3+2]*a[k*3+2]);
                }
                break;
            case AS_AS_POW:
                for (k=0; k<chunk; k++) {res[k] = pow(a[k], b[k]);}
                break;
            case AS_S_POW:
                if (d_stack[p_d-1] == 2.0) {
                    for (k=0; k<chunk; k++) {
                        res[k] = a[k] * a[k];
                    }
                } else if (d_stack[p_d-1] == 3.0) {
                    for (k=0; k<chunk; k++) {
                        res[k] = a[k] * a[k] * a[k];
                    }
                } else {
                    for (k=0; k<chunk; k++) {
                        res[k] = pow(a[k], d_stack[p_d-1]);
                    }
                }
                p_d--;
                break;
            case S_AS_POW:
                for (k=0; k<chunk; k++) {
                    res[k] = pow(d_stack[p_d-1], b[k]);
                }
                p_d--;
                break;
            case S_S_POW:
                d_stack[p_d-2] = pow(d_stack[p_d-2],
                                           d_stack[p_d-1]);
                p_d--;
                break;
            case V_S_POW:
                d_stack[p_d-4] = pow(d_stack[p_d-4],
                                           d_stack[p_d-1]);
                d_stack[p_d-3] = pow(d_stack[p_d-3],
                                           d_stack[p_d-1]);
                d_stack[p_d-2] = pow(d_stack[p_d-2],
                                           d_stack[p_d-1]);
                p_d--;
                break;
            case AV_S_POW:
                if (d_stack[p_d-1]==2.0) {
                    for (k=0; k<chunk; k++) {
                        res[3*k]   = a[3*k]* a[3*k];
                        res[3*k+1] = a[3*k+1]*a[3*k+1];
                        res[3*k+2] = a[3*k+2] * a[3*k+2];
                    }
                } else {
                    for (k=0; k<chunk; k++) {
                        res[3*k] = pow(a[3*k], d_stack[p_d-1]);
                        res[3*k+1] = pow(a[3*k+1], d_stack[p_d-1]);
                        res[3*k+2] = pow(a[3*k+2], d_stack[p_d-1]);
                    }
                }
                p_d--;
                break;
            case S_NEGATE:
                d_stack[p_d-1] = -d_stack[p_d-1];
                break;
            case V_NEGATE:
                d_stack[p_d-3] = -d_stack[p_d-3];
                d_stack[p_d-2] = -d_stack[p_d-2];
                d_stack[p_d-1] = -d_stack[p_d-1];
                break;
            case AS_NEGATE:
                for (k=0; k<chunk; k++) { res[k] = -a[k]; }
                break;
            case AV_NEGATE:
                for (k=0; k<chunk; k++) {
                    res[3*k  ] = -a[3*k];
                    res[3*k+1] = -a[3*k+1];
                    res[3*k+2] = -a[3*k+2];
                }
                break;

            default:
                printf("%ld %ld\n", op, op & ~HEAP_MASK);
                INVALID;
            }
        }
    }

    if (!(p_al == 0) || !(p_as == 0) || !(p_av==0)) {
        PyErr_SetString(PyExc_ValueError, "stack corruption.");
        return 0;
    }
    return 1;

}

static PyObject *array_vm_eval(PyObject *self, PyObject *args)
{
    PyObject *opcodes=0;
    PyObject *double_literals=0;
    PyObject *array_literals=0;
    PyObject *target=0;
    double *c_double_literals=0;
    int nops=0;
    int i;
    long *c_opcodes=0;
    double *c_target=0;
    int array_size=0;
    int outside_loops=0;
    int final_chunk=0;
    if (!PyArg_ParseTuple(args, "OOOO", &opcodes, &double_literals,
                          &array_literals, &target)) return NULL;
    if ( (! PyArray_CheckExact(opcodes)) ||
         (! PyArray_ISINTEGER((PyArrayObject *)opcodes))  ||
         (! PyArray_IS_C_CONTIGUOUS((PyArrayObject *)opcodes)) ||
         (! PyArray_NDIM((PyArrayObject *)opcodes) == 1)) {
        PyErr_SetString(PyExc_ValueError, "opcodes should be 1d contiguous array of type int");
        return NULL;
    }
    nops = PyArray_DIM((PyArrayObject *)opcodes, 0);
    c_opcodes = (long *)((PyArrayObject *)opcodes)->data;

    if ( (! PyArray_CheckExact(double_literals)) ||
         (! PyArray_ISFLOAT((PyArrayObject *)double_literals))  ||
         (! PyArray_IS_C_CONTIGUOUS((PyArrayObject *)double_literals)) ||
         (! PyArray_NDIM((PyArrayObject *)double_literals) == 1)) {
        PyErr_SetString(PyExc_ValueError, "double_literals should be 1d contiguous array of type float");
        return NULL;
    }

    if ( (! PyArray_CheckExact(target)) ||
         (! PyArray_ISFLOAT((PyArrayObject *)target))  ||
         (! PyArray_IS_C_CONTIGUOUS((PyArrayObject *)target)) ||
         (! PyArray_NDIM((PyArrayObject *)target) == 1)) {
        PyErr_SetString(PyExc_ValueError, "target should be 1d contiguous array of type float");
        return NULL;
    }
    array_size = PyArray_DIM((PyArrayObject *)target, 0);
    c_target = PyArray_DATA((PyArrayObject *)target);

    outside_loops = array_size/CHUNK_SIZE;
    final_chunk = array_size % CHUNK_SIZE;

    if (!PyTuple_Check(array_literals)) {
        PyErr_SetString(PyExc_ValueError, "array_literals should be a tuple of contiguous arrays of type float, all the same shape as target.");
        return NULL;
    }

    c_double_literals = (double *)PyArray_DATA((PyArrayObject *) double_literals);
    //#pragma omp parallel for
    for (i=0; i<outside_loops+1; i++) {
        int chunk = CHUNK_SIZE;
        if (i==outside_loops)
            chunk = final_chunk;
        if (! process_chunk(i, chunk, nops, c_double_literals,
                            c_target, c_opcodes, array_literals)) return NULL;
    }
    // what to do if this in not an array expression??
    Py_RETURN_NONE;
}

static PyObject *call_test(PyObject *self, PyObject *args)
{
    int size=0;
    int i=0;
    PyArrayObject *ar0;
    PyArrayObject *ar1;
    PyArrayObject *ar2;

    double *data0;
    double *data1;
    double *data2;


    PyObject *par = PyTuple_GetItem(args, 0);
    ar0 = (PyArrayObject *)PyArray_FROMANY(par, PyArray_DOUBLE, 1, 2, NPY_IN_ARRAY);
    par = PyTuple_GetItem(args, 1);
    ar1 = (PyArrayObject *)PyArray_FROMANY(par, PyArray_DOUBLE, 1, 2, NPY_IN_ARRAY);
    par = PyTuple_GetItem(args, 2);
    ar2 = (PyArrayObject *)PyArray_FROMANY(par, PyArray_DOUBLE, 1, 2, NPY_IN_ARRAY);

    data0 = (double *)ar0->data;
    data1 = (double *)ar1->data;
    data2 = (double *)ar2->data;

    size = PyArray_DIM(ar0,0);
    for (i=0; i<size; i++) {
        data2[i] = data0[i] + data1[i];
    }

    Py_RETURN_NONE;
}

static PyMethodDef
module_functions[] = {
    { "array_vm_eval", array_vm_eval, METH_VARARGS,
      "Array capable VM evaluator" },
    { "call_test", call_test, METH_VARARGS, "test entry point" },
    { NULL }
};

void
initaeev(void)
{
    Py_InitModule3("aeev", module_functions,
                   "Array expression evaluation experiments");
    import_array();
}
