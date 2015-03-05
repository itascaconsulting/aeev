#include "Python.h"
#include "numpy/noprefix.h"
#include "stdio.h"
#include "ops.h"

double eval_double(PyObject *cell, int index)
{
    int op_code = PyInt_AS_LONG(PyTuple_GET_ITEM(cell,0));
    switch (op_code){

    case AS_AS_ADD: case AS_S_ADD: case S_AS_ADD: case S_S_ADD:
        return eval_double(PyTuple_GET_ITEM(cell, 1), index) +
               eval_double(PyTuple_GET_ITEM(cell, 2), index);

    case AS_AS_SUB: case AS_S_SUB: case S_AS_SUB: case S_S_SUB:
        return eval_double(PyTuple_GET_ITEM(cell, 1), index) -
               eval_double(PyTuple_GET_ITEM(cell, 2), index);

    case AS_AS_MUL: case AS_S_MUL: case S_AS_MUL: case S_S_MUL:
        return eval_double(PyTuple_GET_ITEM(cell, 1), index) *
               eval_double(PyTuple_GET_ITEM(cell, 2), index);

    case AS_AS_DIV: case AS_S_DIV: case S_AS_DIV: case S_S_DIV:
        return eval_double(PyTuple_GET_ITEM(cell, 1), index) /
               eval_double(PyTuple_GET_ITEM(cell, 2), index);

    case AS_AS_POW: case AS_S_POW: case S_AS_POW: case S_S_POW:
        return pow(eval_double(PyTuple_GET_ITEM(cell, 1), index),
                   eval_double(PyTuple_GET_ITEM(cell, 2), index));

    case AS_NEGATE: case S_NEGATE:
        return -eval_double(PyTuple_GET_ITEM(cell, 1), index);

    case LIT_S:
        return PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(cell, 1));

    case LIT_AS:
        return ((double *)((PyArrayObject *)
                           PyTuple_GET_ITEM(cell, 1))->data)[index];

    default:
        printf("got %i \n", op_code);
        PyErr_SetString(PyExc_ValueError, "unknown opcode");
        return 0.0;
    }
}

static PyObject *eval(PyObject *self, PyObject *args)
{
    // input is a tuple (opcode)
    PyObject *cell=0;
    if (!PyArg_ParseTuple(args, "O", &cell))
        return NULL;
    if (!PyTuple_Check(cell)) {
        PyErr_SetString(PyExc_ValueError,
                        "expected tuple");
        return NULL;
    }
    return PyFloat_FromDouble(eval_double(cell, 0));
}

static PyObject *array_eval(PyObject *self, PyObject *args)
{
    // input is a tuple (opcode)
    PyObject *cell=0;
    PyObject *target=0;
    int size=0;
    int i=0;
    PyArrayObject *ar;
    if (!PyArg_ParseTuple(args, "OO", &cell, &target))
        return NULL;
    ar = (PyArrayObject *)PyArray_FROMANY(target,
                                          PyArray_DOUBLE,
                                          1,
                                          2,
                                          NPY_IN_ARRAY);
    if (! ar) {
        PyErr_SetString(PyExc_ValueError, "target array error");
        return NULL;
    }
    if (!PyTuple_Check(cell)) {
        PyErr_SetString(PyExc_ValueError, "expected tuple");
        return NULL;
    }
    size = PyArray_DIM(ar,0);
    for (i=0; i<size; i++){
        ((double *)ar->data)[i] = eval_double(cell, i);
    }
    Py_RETURN_NONE;
}

#define CHUNK_SIZE 256
#define GET_HEAP_PTR(arg) (double *) PyArray_DATA((PyArrayObject *) PyTuple_GET_ITEM(array_literals, alstack[arg]))
#define INVALID PyErr_SetString(PyExc_ValueError, "invalid bytecode"); return NULL;
#define STACK_DEPTH 12
#include "makeop.h"
static PyObject *array_vm_eval(PyObject *self, PyObject *args)
{
    PyObject *opcodes=0;
    PyObject *double_literals=0;
    PyObject *array_literals=0;
    PyObject *target=0;
    double *c_double_literals=0;
    int alstack[STACK_DEPTH];
    double astack[STACK_DEPTH][CHUNK_SIZE];
    double dstack[STACK_DEPTH];
    int dstack_ptr=0;
    int astack_ptr=0;
    int alstack_ptr=0;
    int nops=0;
    int i,j,k;
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
    for (i=0; i<outside_loops+1; i++) {
        if (alstack_ptr < 0 || astack_ptr < 0) {
            PyErr_SetString(PyExc_ValueError, "stack corruption.");
            return NULL;
        }

        int chunk = CHUNK_SIZE;
        if (i==outside_loops)
            chunk = final_chunk;
        for (j=0; j<nops; j++) {
            long op = c_opcodes[j];
            if ((op &~ OP_MASK) == LIT_S) {
                dstack[dstack_ptr] = c_double_literals[op & ~BYTECODE_MASK];
                dstack_ptr++;
            }
            else if ((op &~ OP_MASK) == LIT_V) {
                dstack[dstack_ptr] = c_double_literals[op & ~BYTECODE_MASK];
                dstack_ptr++;
                dstack[dstack_ptr] = c_double_literals[(op & ~BYTECODE_MASK)+1];
                dstack_ptr++;
                dstack[dstack_ptr] = c_double_literals[(op & ~BYTECODE_MASK)+2];
                dstack_ptr++;
            }
            else if (((op &~ OP_MASK) == LIT_AS) ||
                     ((op &~ OP_MASK) == LIT_AV) ) {
                alstack[alstack_ptr] = op & ~BYTECODE_MASK;
                alstack_ptr++;
            }
            else  // normal op
            {
                double *res=0; // array result
                double *a = 0; // left
                double *b = 0; // right
                int case_code=0;
                // adapter for testing -- refactor bit order?
                if ((op & A_AV) == A_AV) case_code      |= 1 << 6;
                else if (op & A_AS) case_code           |= 1 << 4;
                if ((op & B_AV) == B_AV) case_code      |= 1 << 5;
                else if (op & B_AS) case_code           |= 1 << 3;
                if (op & A_ON_HEAP) case_code           |= 1 << 2;
                if (op & B_ON_HEAP) case_code           |= 1 << 1;
                if (op & RESULT_TO_HEAP) case_code      |= 1 << 0;

                switch (case_code) {
                    case 0: // 00000 scalar scalar op
                        break;
                    /* case 1: // 00001 */
                    /* case 2: // 00010 */
                    /* case 3: // 00011 */
                    /* case 4: // 00100 */
                    /* case 5: // 00101 */
                    /* case 6: // 00110 */
                    /* case 7: // 00111 */
                    /*     INVALID; */
                case 8: // 01000 a-scalar b-array-stack, r-stack
                    res = astack[astack_ptr-1];
                    b = astack[astack_ptr-1];
                    break;
                case 9: // 01001 a-scalar b-array-stach, r-heap
                    res = c_target + i * CHUNK_SIZE;
                    b = astack[astack_ptr-1];
                    break;
                case 10: // 01010 a-scalar b-array-heap, r-stack
                    res = astack[astack_ptr];
                    b = GET_HEAP_PTR(alstack_ptr-1) + i * CHUNK_SIZE;
                    astack_ptr++;
                    alstack_ptr--;
                    break;
                case 11: // 01011 a-scalar, b-array-heap, r-heap
                    res = c_target + i * CHUNK_SIZE;
                    b = GET_HEAP_PTR(alstack_ptr-1) + i * CHUNK_SIZE;
                    alstack_ptr--;
                    break;
                    /* case 12: // 01100 */
                    /* case 13: // 01101 */
                    /* case 14: // 01110 */
                    /* case 15: // 01111 */
                    /*     INVALID; */
                case 16: // 10000 a-array-stack, b-scalar, r-stack
                    res = astack[astack_ptr-1];
                    a = astack[astack_ptr-1];
                    break;
                case 17: // 10001 a-array-stack, b-scalar, r-heap
                    res = c_target + i * CHUNK_SIZE;
                    a = astack[astack_ptr-1];
                    astack_ptr--;
                    break;
                case 18: // 10010
                    //case 19: // 10011                    INVALID;
                case 20: // 10100 a-array-heap, b-scalar, r-stack
                    res = astack[astack_ptr];
                    a = GET_HEAP_PTR(alstack_ptr-1) + i * CHUNK_SIZE;
                    alstack_ptr--;
                    astack_ptr++;
                    break;
                case 21: // 10101 a-array-heap, b-scalar, r-heap
                    res = c_target + i * CHUNK_SIZE;
                    a = GET_HEAP_PTR(alstack_ptr-1) + i * CHUNK_SIZE;
                    alstack_ptr--;
                    break;
                    /* case 22: // 10110 */
                    /* case 23: // 10111 */
                    /*     INVALID; */
                case 24: // 11000 a-stack, b-stack, r-stack
                    res = astack[astack_ptr-2];
                    a = astack[astack_ptr-2];
                    b = astack[astack_ptr-1];
                    astack_ptr--;
                    break;
                case 25: // 11001 a-stack, b-stack, r-heap
                    res = c_target + i * CHUNK_SIZE;
                    a = astack[astack_ptr-2];
                    b = astack[astack_ptr-1];
                    astack_ptr -= 2;
                    break;
                case 26: // 11010 a-stack, b-heap, r-stack
                    res = astack[astack_ptr-1];
                    a = astack[astack_ptr-1];
                    b = GET_HEAP_PTR(alstack_ptr-1) + i * CHUNK_SIZE;
                    alstack_ptr--;
                    break;
                case 27: // 11011 a-stack, b-heap, r-heap
                    res = c_target + i * CHUNK_SIZE;
                    a = astack[astack_ptr-1];
                    b = GET_HEAP_PTR(alstack_ptr-1) + i * CHUNK_SIZE;
                    astack_ptr--;
                    alstack_ptr--;
                    break;
                case 28: // 11100  a-heap, b-stack, r-stack
                    res = astack[astack_ptr-1];
                    a = GET_HEAP_PTR(alstack_ptr-1) + i * CHUNK_SIZE;
                    b = astack[astack_ptr-1];
                    alstack_ptr--;
                    break;
                case 29: //11101 a-heap, b-stack, r-heap
                    res = c_target + i * CHUNK_SIZE;
                    a = GET_HEAP_PTR(alstack_ptr-1) + i * CHUNK_SIZE;
                    b = astack[astack_ptr-1];
                    alstack_ptr--;
                    astack_ptr--;
                    break;
                case 30: // 11110 a-heap, b-heap, rstack
                    res = astack[astack_ptr];
                    a = GET_HEAP_PTR(alstack_ptr-2) + i * CHUNK_SIZE;
                    b = GET_HEAP_PTR(alstack_ptr-1) + i * CHUNK_SIZE;
                    astack_ptr++;
                    alstack_ptr -= 2;
                    break;
                case 31: // 11111 a-heap. b-heap, r-heap
                    res = c_target + i * CHUNK_SIZE;
                    a = GET_HEAP_PTR(alstack_ptr-2) + i * CHUNK_SIZE;
                    b = GET_HEAP_PTR(alstack_ptr-1) + i * CHUNK_SIZE;
                    alstack_ptr -= 2;
                    break;

                case  32: // 0100000 a-s, b-av, a-stack, b-stack, r-stack
                    res = astack[astack_ptr-3];
                    b = astack[astack_ptr-3];
                    break;
                case  33: // 0100001 a-s, b-av, a-stack, b-stack, r-heap
                    res = c_target + 3*i * CHUNK_SIZE;
                    b = astack[astack_ptr-3];
                    astack_ptr -= 3;
                    break;
                case  34: // 0100010 a-s, b-av, a-stack, b-heap, r-stack
                    res = astack[astack_ptr];
                    b = GET_HEAP_PTR(alstack_ptr-1) + 3*i * CHUNK_SIZE;
                    astack_ptr += 3;
                    alstack_ptr--;
                    break;
                case  35: // 0100011 a-s, b-av, a-stack, b-heap, r-heap
                    res = c_target + 3*i * CHUNK_SIZE;
                    b = GET_HEAP_PTR(alstack_ptr-1) + 3*i * CHUNK_SIZE;
                    alstack_ptr--;
                    break;
                case  36: // 0100100 a-s, b-av, a-heap, b-stack, r-stack
                case  37: // 0100101 a-s, b-av, a-heap, b-stack, r-heap
                case  38: // 0100110 a-s, b-av, a-heap, b-heap, r-stack
                case  39: // 0100111 a-s, b-av, a-heap, b-heap, r-heap
                case  40: // 0101000 a-s, b-av
                case  41: // 0101001
                case  42: // 0101010
                case  43: // 0101011
                case  44: // 0101100
                case  45: // 0101101
                case  46: // 0101110
                case  47: // 0101111
                    INVALID;

                case  48: // 0110000 a-as, b-av, a-stack, b-stack, r-stack
                    res = astack[astack_ptr-4];
                    for (k=0; k<chunk; k++) { // copy a to bottom of stack
                        astack[astack_ptr][k] = astack[astack_ptr-4][k];
                    }
                    a = astack[astack_ptr];
                    b = astack[astack_ptr-3];
                    astack_ptr -= 1;
                    break;
                case  49: // 0110001  a-as, b-av, a-stack, b-stack, r-heap
                    res = c_target + 3*i * CHUNK_SIZE;
                    a = astack[astack_ptr-4];
                    b = astack[astack_ptr-3];
                    astack_ptr -= 4;
                    break;
                case  50: // 0110010  a-as, b-av, a-stack, b-heap, r-stack
                    res = astack[astack_ptr-1];
                    for (k=0; k<chunk; k++) { // copy a to bottom of stack
                        astack[astack_ptr+2][k] = astack[astack_ptr-1][k];
                    }
                    a = astack[astack_ptr+2];
                    b = GET_HEAP_PTR(alstack_ptr-1) + 3*i * CHUNK_SIZE;
                    alstack_ptr--;
                    astack_ptr+=2;
                    break;
                case  51: // 0110011  a-as, b-av, a-stack, b-heap, r-heap
                    res = c_target + 3*i * CHUNK_SIZE;
                    a = astack[astack_ptr-1];
                    b = GET_HEAP_PTR(alstack_ptr-1) + 3*i * CHUNK_SIZE;
                    alstack_ptr--;
                    astack_ptr--;
                    break;
                case  52: // 0110100 a-as, b-av, a-heap, b-stack, r-stack
                    res = astack[astack_ptr-3];
                    a = GET_HEAP_PTR(alstack_ptr-1) + i * CHUNK_SIZE;
                    b = astack[astack_ptr-3];
                    alstack_ptr -= 1;
                    break;
                case  53: // 0110101 a-as, b-av, a-heap, b-stack, r-heap
                    res = c_target + 3*i * CHUNK_SIZE;
                    a = GET_HEAP_PTR(alstack_ptr-1) + i * CHUNK_SIZE;
                    b = astack[astack_ptr-3];
                    alstack_ptr -= 1;
                    astack_ptr -= 3;
                    break;
                case  54: // 0110110 a-as, b-av, a-heap, b-heap, r-stack
                    res = astack[astack_ptr];
                    a = GET_HEAP_PTR(alstack_ptr-2) + i * CHUNK_SIZE;
                    b = GET_HEAP_PTR(alstack_ptr-1) + 3*i * CHUNK_SIZE;
                    alstack_ptr -= 2;
                    astack_ptr += 3;
                    break;
                case  55: // 0110111 a-as, b-av, a-heap, b-heap, r-heap
                    res = c_target + 3*i * CHUNK_SIZE;
                    a = GET_HEAP_PTR(alstack_ptr-2) + i * CHUNK_SIZE;
                    b = GET_HEAP_PTR(alstack_ptr-1) + 3*i * CHUNK_SIZE;
                    alstack_ptr -= 2;
                    break;

                case  56: // 0111000
                case  57: // 0111001
                case  58: // 0111010
                case  59: // 0111011
                case  60: // 0111100
                case  61: // 0111101
                case  62: // 0111110
                case  63: // 0111111
                    INVALID;

                case  64: // 1000000 a-av b-s a-stack b-stack r-stack
                    res = astack[astack_ptr-3];
                    a = astack[astack_ptr-3];
                    break;
                case  65: // 1000001 a-av b-s a-stack b-stack r-heap
                    res = c_target + 3*i * CHUNK_SIZE;
                    a = astack[astack_ptr-3];
                    astack_ptr -= 3;
                    break;
                case  66: // 1000010
                case  67: // 1000011
                    INVALID;
                case  68: // 1000100 a-av, b-s, a-heap, r-stack
                    res = astack[astack_ptr];
                    a = GET_HEAP_PTR(alstack_ptr-1) + 3*i * CHUNK_SIZE;
                    astack_ptr += 3;
                    alstack_ptr--;
                    break;
                case  69: // 1000101 a-av, b-s, a-heap, r-heap
                    res = c_target + 3*i * CHUNK_SIZE;
                    a = GET_HEAP_PTR(alstack_ptr-1) + 3*i * CHUNK_SIZE;
                    alstack_ptr--;
                    break;
                case  70: // 1000110
                case  71: // 1000111
                    INVALID;
                case  72: // 1001000 a-av b-as a-stack b-stack r-stack
                    res = astack[astack_ptr-4];
                    a = astack[astack_ptr-4];
                    b = astack[astack_ptr-1];
                    astack_ptr--;
                    break;
                case  73: // 1001001 a-av b-as a-stack b-stack r-heap
                    res = c_target + 3*i * CHUNK_SIZE;
                    a = astack[astack_ptr-4];
                    b = astack[astack_ptr-1];
                    astack_ptr -= 4;
                    break;
                case  74: // 1001010 a-av b-as a-stack b-heap r-stack
                    res = astack[astack_ptr-3];
                    a = astack[astack_ptr-3];
                    b = GET_HEAP_PTR(alstack_ptr-1) + i * CHUNK_SIZE;
                    alstack_ptr--;
                    break;
                case  75: // 1001011 a-av b-as a-stack b-heap r-heap
                    res = c_target + 3*i * CHUNK_SIZE;
                    a = astack[astack_ptr-3];
                    b = GET_HEAP_PTR(alstack_ptr-1) + i * CHUNK_SIZE;
                    alstack_ptr--;
                    break;
                case  76: // 1001100 a-av b-as a-heap b-stack r-stack
                    res = astack[astack_ptr-1];
                    a = GET_HEAP_PTR(alstack_ptr-1) + 3*i * CHUNK_SIZE;
                    for (k=0; k<chunk; k++) {
                        astack[astack_ptr+2][k] = astack[astack_ptr-1][k];
                    }
                    b = astack[astack_ptr+2];
                    alstack_ptr--;
                    astack_ptr += 2;
                    break;
                case  77: // 1001101 a-av b-as a-heap b-stack r-heap
                    res = c_target + 3*i * CHUNK_SIZE;
                    a = GET_HEAP_PTR(alstack_ptr-1) + 3*i * CHUNK_SIZE;
                    b = astack[astack_ptr-1];
                    astack_ptr--;
                    alstack_ptr--;
                    break;
                case  78: // 1001110 a-av b-as a-heap b-heap r-stack
                    res = astack[astack_ptr];
                    a = GET_HEAP_PTR(alstack_ptr-2) + 3*i * CHUNK_SIZE;
                    b = GET_HEAP_PTR(alstack_ptr-1) + i * CHUNK_SIZE;
                    alstack_ptr -= 2;
                    astack_ptr += 3;
                    break;
                case  79: // 1001111 a-av b-as a-heap b-heap r-heap
                    res = c_target + 3*i * CHUNK_SIZE;
                    a = GET_HEAP_PTR(alstack_ptr-2) + 3*i * CHUNK_SIZE;
                    b = GET_HEAP_PTR(alstack_ptr-1) + i * CHUNK_SIZE;
                    alstack_ptr -= 2;
                    break;

                case  80: // 1010000
                case  81: // 1010001
                case  82: // 1010010
                case  83: // 1010011
                case  84: // 1010100
                case  85: // 1010101
                case  86: // 1010110
                case  87: // 1010111
                case  88: // 1011000
                case  89: // 1011001
                case  90: // 1011010
                case  91: // 1011011
                case  92: // 1011100
                case  93: // 1011101
                case  94: // 1011110
                case  95: // 1011111
                    INVALID;

// a-av b-av a-as b-as a-heap b-heap r-heap
                case  96: // 1100000 a-av b-av a-stack b-stack r-stack
                    res = astack[astack_ptr-6];
                    a = astack[astack_ptr-6];
                    b = astack[astack_ptr-3];
                    astack_ptr -= 3;
                    break;
                case  97: // 1100001 a-av b-av a-stack b-stack r-stack
                    res = c_target + 3*i * CHUNK_SIZE;
                    a = astack[astack_ptr-6];
                    b = astack[astack_ptr-3];
                    astack_ptr -= 6;
                    break;
                case  98: // 1100010 a-av b-av a-stack b-heap r-stack
                    res = astack[astack_ptr-3];
                    a = astack[astack_ptr-3];
                    b = GET_HEAP_PTR(alstack_ptr-1) + 3*i * CHUNK_SIZE;
                    alstack_ptr--;
                    break;
                case  99: // 1100011 a-av b-av a-stack b-heap r-heap
                    res = c_target + 3*i * CHUNK_SIZE;
                    a = astack[astack_ptr-3];
                    b = GET_HEAP_PTR(alstack_ptr-1) + 3*i * CHUNK_SIZE;
                    alstack_ptr--;
                    astack_ptr -= 3;
                    break;
                case 100: // 1100100 a-av b-av a-heap b-stack r-stack
                    res = astack[astack_ptr-3];
                    a = GET_HEAP_PTR(alstack_ptr-1) + 3*i * CHUNK_SIZE;
                    b  = astack[astack_ptr-3];
                    alstack_ptr -= 1;
                    break;
                case 101: // 1100101 a-av b-av a-heap b-stack r-heap
                    res = c_target + 3*i * CHUNK_SIZE;
                    a = GET_HEAP_PTR(alstack_ptr-1) + 3*i * CHUNK_SIZE;
                    b  = astack[astack_ptr-3];
                    alstack_ptr -= 1;
                    astack_ptr -= 3;
                    break;
                case 102: // 1100110 a-av b-av a-heap b-heap r-stack
                    res = astack[astack_ptr];
                    a = GET_HEAP_PTR(alstack_ptr-2) + 3*i * CHUNK_SIZE;
                    b = GET_HEAP_PTR(alstack_ptr-1) + 3*i * CHUNK_SIZE;
                    astack_ptr += 3;
                    alstack_ptr -= 2;
                    break;
                case 103: // 1100111 a-av b-av a-heap b-heap r-heap
                    res = c_target + 3*i * CHUNK_SIZE;
                    a = GET_HEAP_PTR(alstack_ptr-2) + 3*i * CHUNK_SIZE;
                    b = GET_HEAP_PTR(alstack_ptr-1) + 3*i * CHUNK_SIZE;
                    alstack_ptr -= 2;
                    break;

                default:
                    INVALID;
                }

                if (alstack_ptr >= STACK_DEPTH ||
                    astack_ptr >= STACK_DEPTH) {
                    PyErr_SetString(PyExc_ValueError, "Stack overflow.");
                    return NULL;
                }

                //printf("opcode %i %i %i\n", op, op &~BYTECODE_MASK, case_code);
                switch (op & ~HEAP_MASK) {
                    OPERATOR(ADD, +);
                    OPERATOR(SUB, -);
                    OPERATOR(MUL, *);
                    OPERATOR(DIV, /);

                case AS_AS_POW:
                    for (k=0; k<chunk; k++) {res[k] = pow(a[k], b[k]);}
                    break;
                case AS_S_POW:
                    if (dstack[dstack_ptr-1] == 2.0) {
                        for (k=0; k<chunk; k++) {
                            res[k] = a[k] * a[k];
                        }
                    } else if (dstack[dstack_ptr-1] == 3.0) {
                        for (k=0; k<chunk; k++) {
                            res[k] = a[k] * a[k] * a[k];
                        }
                    } else {
                        for (k=0; k<chunk; k++) {
                            res[k] = pow(a[k], dstack[dstack_ptr-1]);
                        }
                    }
                    dstack_ptr--;
                    break;
                case S_AS_POW:
                    for (k=0; k<chunk; k++) {
                        res[k] = pow(dstack[dstack_ptr-1], b[k]);
                    }
                    dstack_ptr--;
                    break;
                case S_S_POW:
                    dstack[dstack_ptr-2] = pow(dstack[dstack_ptr-2],
                                               dstack[dstack_ptr-1]);
                    dstack_ptr--;
                    break;

                case V_S_POW:
                    dstack[dstack_ptr-4] = pow(dstack[dstack_ptr-4],
                                               dstack[dstack_ptr-1]);
                    dstack[dstack_ptr-3] = pow(dstack[dstack_ptr-3],
                                               dstack[dstack_ptr-1]);
                    dstack[dstack_ptr-2] = pow(dstack[dstack_ptr-2],
                                               dstack[dstack_ptr-1]);
                    dstack_ptr--;
                    break;

                case AV_S_POW:
                    if (dstack[dstack_ptr-1]==2.0) {
                        for (k=0; k<chunk; k++) {
                            res[3*k]   = a[3*k]* a[3*k];
                            res[3*k+1] = a[3*k+1]*a[3*k+1];
                            res[3*k+2] = a[3*k+2] * a[3*k+2];
                        }
                    } else {
                        for (k=0; k<chunk; k++) {
                            res[3*k] = pow(a[3*k], dstack[dstack_ptr-1]);
                            res[3*k+1] = pow(a[3*k+1], dstack[dstack_ptr-1]);
                            res[3*k+2] = pow(a[3*k+2], dstack[dstack_ptr-1]);
                        }
                    }
                    dstack_ptr--;
                    break;
                case S_NEGATE:
                    dstack[dstack_ptr-1] = -dstack[dstack_ptr-1];
                    break;
                case V_NEGATE:
                    dstack[dstack_ptr-3] = -dstack[dstack_ptr-3];
                    dstack[dstack_ptr-2] = -dstack[dstack_ptr-2];
                    dstack[dstack_ptr-1] = -dstack[dstack_ptr-1];
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
                    printf("%i %i\n", op, op & ~HEAP_MASK);
                    INVALID;
                }
            }
        }
    }
    // what to do if this in not an array expression??
    Py_RETURN_NONE;
}

static PyObject *vm_eval(PyObject *self, PyObject *args)
{
    // input is a tuple (opcode)
    PyObject *ops=0;
    PyObject *literals=0;
    int c_op=0;
    int n_opt=0;
    int stack_pointer = 0; // points to available stack location
    double stack[32];
    int i;

    if (!PyArg_ParseTuple(args, "OO", &ops, &literals))
        return NULL;
    if (! (PyTuple_Check(ops) && PyTuple_Check(literals)))
        return NULL;
    n_opt = PyTuple_GET_SIZE(ops);
    for (i=0; i<n_opt; i++) {
        c_op = PyInt_AS_LONG(PyTuple_GET_ITEM(ops, i));
        if ((c_op &~ OP_MASK) == LIT_S) {
            stack[stack_pointer] = PyFloat_AS_DOUBLE(
                PyTuple_GET_ITEM(literals, c_op & ~BYTECODE_MASK));
            stack_pointer++;
        }
        else if ((c_op &~ OP_MASK) == LIT_V) {
            stack[stack_pointer] = PyFloat_AS_DOUBLE(
                PyTuple_GET_ITEM(literals, c_op & ~BYTECODE_MASK));
            stack_pointer++;
            stack[stack_pointer] = PyFloat_AS_DOUBLE(
                PyTuple_GET_ITEM(literals, (c_op & ~BYTECODE_MASK)+1));
            stack_pointer++;
            stack[stack_pointer] = PyFloat_AS_DOUBLE(
                PyTuple_GET_ITEM(literals, (c_op & ~BYTECODE_MASK)+2));
            stack_pointer++;
        }
        else {
            switch (c_op) {
            case AS_AS_ADD: case AS_S_ADD: case S_AS_ADD: case S_S_ADD:
                stack[stack_pointer-2] = stack[stack_pointer-2] +
                                         stack[stack_pointer-1];
                stack_pointer--;
                break;
            case AS_AS_POW: case AS_S_POW: case S_AS_POW: case S_S_POW:
                stack[stack_pointer-2] = pow(stack[stack_pointer-2],
                                             stack[stack_pointer-1]);
                stack_pointer--;
                break;
            case V_S_ADD:
                stack[stack_pointer-4] = stack[stack_pointer-4] + \
                    stack[stack_pointer-1];
                stack[stack_pointer-3] = stack[stack_pointer-3] + \
                    stack[stack_pointer-1];
                stack[stack_pointer-2] = stack[stack_pointer-2] + \
                    stack[stack_pointer-1];
                stack_pointer--;
                break;

            case V_V_ADD:
                stack[stack_pointer-6] = stack[stack_pointer-6] + \
                    stack[stack_pointer-3];
                stack[stack_pointer-5] = stack[stack_pointer-5] + \
                    stack[stack_pointer-2];
                stack[stack_pointer-4] = stack[stack_pointer-4] + \
                    stack[stack_pointer-1];
                stack_pointer -= 3;
                break;

            default:
                PyErr_SetString(PyExc_ValueError,
                                "unknown opcode");
                return NULL;
            }
        }
    }
    if (PyInt_AS_LONG(PyTuple_GET_ITEM(ops, 0)) & R_V)
        return Py_BuildValue("(ddd)", stack[0], stack[1], stack[2]);
    return PyFloat_FromDouble(stack[0]);
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


// Module functions table.
static PyMethodDef
module_functions[] = {
    { "eval", eval, METH_VARARGS, "AST walking interpreter" },
    { "array_eval", array_eval, METH_VARARGS,
      "Array capable AST walking interpreter" },
    { "vm_eval", vm_eval, METH_VARARGS, "Virtual machine evaluator" },
    { "array_vm_eval", array_vm_eval, METH_VARARGS,
      "Array capable VM evaluator" },
    { "call_test", call_test, METH_VARARGS, "test entry point" },
    { NULL }
};

// This function is called to initialize the module.

void
initaeev(void)
{
    Py_InitModule3("aeev", module_functions,
                   "Array expression evaluation experiments");
    import_array();
}
