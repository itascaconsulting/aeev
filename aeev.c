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

#define SET_RES_HEAP \
if (av_result) {\
   res = c_target + 3 * i * CHUNK_SIZE;\
} else {\
   res = c_target + i * CHUNK_SIZE;\
}


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
            int case_code=0;
            int av_result=0;
            // adapter for testing -- refactor bit order?
            // bits: (a-av, b-av, a-as, b-as, a-heap, b-heap, r-heap)
            // this 7 bit int describes the types and if the
            // operands and result are on the stack or heap.
            if ((op & A_AV) == A_AV) case_code      |= 1 << 6;
            else if (op & A_AS) case_code           |= 1 << 4;
            if ((op & B_AV) == B_AV) case_code      |= 1 << 5;
            else if (op & B_AS) case_code           |= 1 << 3;
            if (op & A_ON_HEAP) case_code           |= 1 << 2;
            if (op & B_ON_HEAP) case_code           |= 1 << 1;
            if (op & RESULT_TO_HEAP) case_code      |= 1 << 0;
            if ((op & R_AV) == R_AV) av_result = 1;
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
                b = as_stack[p_as-1];
                if (av_result) {
                    res = av_stack[p_av];
                    p_av++;
                    p_as--;
                } else {
                    res = as_stack[p_as-1];
                }
                break;
            case 9: // 01001 a-scalar b-array-stack, r-heap
                SET_RES_HEAP;
                b = as_stack[p_as-1];
                break;
            case 10: // 01010 a-scalar b-array-heap, r-stack
                if (av_result) {
                    res = av_stack[p_av];
                    p_av++;
                } else {
                    res = as_stack[p_as];
                    p_as++;
                }
                b = GET_HEAP_PTR(p_al-1) + i * CHUNK_SIZE;
                p_al--;
                break;
            case 11: // 01011 a-scalar, b-array-heap, r-heap
                SET_RES_HEAP;
                b = GET_HEAP_PTR(p_al-1) + i * CHUNK_SIZE;
                p_al--;
                break;
                /* case 12: // 01100 */
                /* case 13: // 01101 */
                /* case 14: // 01110 */
                /* case 15: // 01111 */
                /*     INVALID; */
            case 16: // 10000 a-array-stack, b-scalar, r-stack
                a = as_stack[p_as-1];
                if (av_result) {
                    res = av_stack[p_av];
                    p_av++;
                    p_as--;
                } else {
                    res = as_stack[p_as-1];
                }
                break;
            case 17: // 10001 a-array-stack, b-scalar, r-heap
                SET_RES_HEAP;
                a = as_stack[p_as-1];
                p_as--;
                break;
            case 18: // 10010
                //case 19: // 10011                    INVALID;
            case 20: // 10100 a-array-heap, b-scalar, r-stack
                if (av_result) {
                    res = av_stack[p_av];
                    p_av++;
                } else {
                    res = as_stack[p_as];
                    p_as++;
                }
                a = GET_HEAP_PTR(p_al-1) + i * CHUNK_SIZE;
                p_al--;
                break;
            case 21: // 10101 a-array-heap, b-scalar, r-heap
                SET_RES_HEAP;
                a = GET_HEAP_PTR(p_al-1) + i * CHUNK_SIZE;
                p_al--;
                break;
                /* case 22: // 10110 */
                /* case 23: // 10111 */
                /*     INVALID; */
            case 24: // 11000 a-stack, b-stack, r-stack
                a = as_stack[p_as-2];
                b = as_stack[p_as-1];
                if (av_result) {
                    res = av_stack[p_av];
                    p_av++;
                    p_as -= 2;
                } else {
                    res = as_stack[p_as-2];
                    p_as--;
                }
                break;
            case 25: // 11001 a-stack, b-stack, r-heap
                SET_RES_HEAP;
                a = as_stack[p_as-2];
                b = as_stack[p_as-1];
                p_as -= 2;
                break;
            case 26: // 11010 a-stack, b-heap, r-stack
                a = as_stack[p_as-1];
                b = GET_HEAP_PTR(p_al-1) + i * CHUNK_SIZE;
                p_al--;
                if (av_result) {
                    res = av_stack[p_av];
                    p_av++;
                    p_as--;
                } else {
                    res = as_stack[p_as-1];
                }
                break;
            case 27: // 11011 a-stack, b-heap, r-heap
                SET_RES_HEAP;
                a = as_stack[p_as-1];
                b = GET_HEAP_PTR(p_al-1) + i * CHUNK_SIZE;
                p_as--;
                p_al--;
                break;
            case 28: // 11100  a-heap, b-stack, r-stack
                a = GET_HEAP_PTR(p_al-1) + i * CHUNK_SIZE;
                b = as_stack[p_as-1];
                p_al--;
                if (av_result) {
                    res = av_stack[p_av];
                    p_av++;
                    p_as--;
                } else {
                    res = as_stack[p_as-1];
                }
                break;
            case 29: //11101 a-heap, b-stack, r-heap
                SET_RES_HEAP;
                a = GET_HEAP_PTR(p_al-1) + i * CHUNK_SIZE;
                b = as_stack[p_as-1];
                p_al--;
                p_as--;
                break;
            case 30: // 11110 a-heap, b-heap, rstack
                a = GET_HEAP_PTR(p_al-2) + i * CHUNK_SIZE;
                b = GET_HEAP_PTR(p_al-1) + i * CHUNK_SIZE;
                p_al -= 2;
                if (av_result) {
                    res = av_stack[p_av];
                    p_av++;
                } else {
                    res = as_stack[p_as];
                    p_as++;
                }
                break;
            case 31: // 11111 a-heap. b-heap, r-heap
                SET_RES_HEAP;
                a = GET_HEAP_PTR(p_al-2) + i * CHUNK_SIZE;
                b = GET_HEAP_PTR(p_al-1) + i * CHUNK_SIZE;
                p_al -= 2;
                break;
            case  32: // 0100000 a-s, b-av, a-stack, b-stack, r-stack
                b = av_stack[p_av-1];
                if (av_result) {
                    res = av_stack[p_av-1];
                } else {
                    res = as_stack[p_as];
                    p_as++;
                }
                break;
            case  33: // 0100001 a-s, b-av, a-stack, b-stack, r-heap
                SET_RES_HEAP;
                b = av_stack[p_av-1];
                p_av--;
                break;
            case  34: // 0100010 a-s, b-av, a-stack, b-heap, r-stack
                b = GET_HEAP_PTR(p_al-1) + 3*i * CHUNK_SIZE;
                p_al--;
                if (av_result) {
                    res = av_stack[p_av];
                    p_av++;
                } else {
                    res = as_stack[p_as];
                    p_as++;
                }
                break;
            case  35: // 0100011 a-s, b-av, a-stack, b-heap, r-heap
                SET_RES_HEAP;
                b = GET_HEAP_PTR(p_al-1) + 3*i * CHUNK_SIZE;
                p_al--;
                break;
                /* case  36: // 0100100 a-s, b-av, a-heap, b-stack, r-stack */
                /* case  37: // 0100101 a-s, b-av, a-heap, b-stack, r-heap */
                /* case  38: // 0100110 a-s, b-av, a-heap, b-heap, r-stack */
                /* case  39: // 0100111 a-s, b-av, a-heap, b-heap, r-heap */
                /* case  40: // 0101000 a-s, b-av */
                /* case  41: // 0101001 */
                /* case  42: // 0101010 */
                /* case  43: // 0101011 */
                /* case  44: // 0101100 */
                /* case  45: // 0101101 */
                /* case  46: // 0101110 */
                /* case  47: // 0101111 */
                /*     INVALID; */
            case  48: // 0110000 a-as, b-av, a-stack, b-stack, r-stack
                a = as_stack[p_as-1];
                b = av_stack[p_av-1];
                if (av_result) {
                    res = av_stack[p_av-1];
                    p_as--;
                } else {
                    res = as_stack[p_as-1];
                    p_av--;
                }
                break;
            case  49: // 0110001  a-as, b-av, a-stack, b-stack, r-heap
                SET_RES_HEAP;
                a = as_stack[p_as-2];
                b = av_stack[p_av-1];
                p_as--;
                p_av--;
                break;
            case  50: // 0110010  a-as, b-av, a-stack, b-heap, r-stack
                a = as_stack[p_as-1];
                b = GET_HEAP_PTR(p_al-1) + 3*i * CHUNK_SIZE;
                p_al--;
                if (av_result) {
                    res = av_stack[p_av];
                    p_av++;
                    p_as--;
                } else {
                    res = as_stack[p_as-1];
                }
                break;
            case  51: // 0110011  a-as, b-av, a-stack, b-heap, r-heap
                SET_RES_HEAP;
                a = as_stack[p_as-1];
                b = GET_HEAP_PTR(p_al-1) + 3*i * CHUNK_SIZE;
                p_al--;
                p_as--;
                break;
            case  52: // 0110100 a-as, b-av, a-heap, b-stack, r-stack
                a = GET_HEAP_PTR(p_al-1) + i * CHUNK_SIZE;
                b = av_stack[p_av-1];
                p_al--;
                if (av_result) {
                    res = av_stack[p_av-1];
                } else {
                    res = as_stack[p_as];
                    p_as++;
                    p_av--;
                }
                break;
            case  53: // 0110101 a-as, b-av, a-heap, b-stack, r-heap
                SET_RES_HEAP;
                a = GET_HEAP_PTR(p_al-1) + i * CHUNK_SIZE;
                b = as_stack[p_as-3];
                p_al -= 1;
                p_as -= 3;
                break;
            case  54: // 0110110 a-as, b-av, a-heap, b-heap, r-stack
                a = GET_HEAP_PTR(p_al-2) + i * CHUNK_SIZE;
                b = GET_HEAP_PTR(p_al-1) + 3*i * CHUNK_SIZE;
                p_al -= 2;
                if (av_result) {
                    res = av_stack[p_av];
                    p_av++;
                } else {
                    res = as_stack[p_as];
                    p_as++;
                }
                break;
            case  55: // 0110111 a-as, b-av, a-heap, b-heap, r-heap
                SET_RES_HEAP
                a = GET_HEAP_PTR(p_al-2) + i * CHUNK_SIZE;
                b = GET_HEAP_PTR(p_al-1) + 3*i * CHUNK_SIZE;
                p_al -= 2;
                break;
                /* case  56: // 0111000 */
                /* case  57: // 0111001 */
                /* case  58: // 0111010 */
                /* case  59: // 0111011 */
                /* case  60: // 0111100 */
                /* case  61: // 0111101 */
                /* case  62: // 0111110 */
                /* case  63: // 0111111 */
                /*     INVALID; */
            case  64: // 1000000 a-av b-s a-stack b-stack r-stack
                a = av_stack[p_av-1];
                if (av_result) {
                    res = av_stack[p_av-1];
                } else {
                    res = as_stack[p_as];
                    p_as++;
                    p_av--;
                }
                break;
            case  65: // 1000001 a-av b-s a-stack b-stack r-heap
                SET_RES_HEAP;
                a = av_stack[p_av-1];
                p_av--;
                break;
                /* case  66: // 1000010 */
                /* case  67: // 1000011 */
                /*     INVALID; */
            case  68: // 1000100 a-av, b-s, a-heap, r-stack
                a = GET_HEAP_PTR(p_al-1) + 3*i * CHUNK_SIZE;
                p_al--;
                if (av_result) {
                    res = av_stack[p_av];
                    p_av++;
                } else {
                    res = as_stack[p_as];
                    p_as++;
                }
                break;
            case  69: // 1000101 a-av, b-s, a-heap, r-heap
                SET_RES_HEAP;
                a = GET_HEAP_PTR(p_al-1) + 3*i * CHUNK_SIZE;
                p_al--;
                break;
                /* case  70: // 1000110 */
                /* case  71: // 1000111 */
                /*     INVALID; */
            case  72: // 1001000 a-av b-as a-stack b-stack r-stack
                a = av_stack[p_av-1];
                b = as_stack[p_as-1];
                if (av_result) {
                    res = av_stack[p_av-1];
                    p_as--;
                } else {
                    res = as_stack[p_as-1];
                    p_av--;
                }
                break;
            case  73: // 1001001 a-av b-as a-stack b-stack r-heap
                SET_RES_HEAP;
                a = av_stack[p_av-1];
                b = as_stack[p_as-1];
                p_av--;
                p_as--;
                break;
            case  74: // 1001010 a-av b-as a-stack b-heap r-stack
                a = av_stack[p_av-1];
                b = GET_HEAP_PTR(p_al-1) + i * CHUNK_SIZE;
                p_al--;
                if (av_result) {
                    res = av_stack[p_av-1];
                } else {
                    res = as_stack[p_as-1];
                    p_av--;
                    p_as++;
                }
                break;
            case  75: // 1001011 a-av b-as a-stack b-heap r-heap
                SET_RES_HEAP;
                a = av_stack[p_av-1];
                p_av--;
                b = GET_HEAP_PTR(p_al-1) + i * CHUNK_SIZE;
                p_al--;
                break;
            case  76: // 1001100 a-av b-as a-heap b-stack r-stack
                a = GET_HEAP_PTR(p_al-1) + 3*i * CHUNK_SIZE;
                p_al--;
                b = as_stack[p_as-1];
                if (av_result) {
                    res = av_stack[p_av];
                    p_av++;
                    p_as--;
                } else {
                    res = as_stack[p_as-1];
                }
                break;
            case  77: // 1001101 a-av b-as a-heap b-stack r-heap
                SET_RES_HEAP
                a = GET_HEAP_PTR(p_al-1) + 3*i * CHUNK_SIZE;
                p_al--;
                b = as_stack[p_as-1];
                p_as--;
                break;
            case  78: // 1001110 a-av b-as a-heap b-heap r-stack
                a = GET_HEAP_PTR(p_al-2) + 3*i * CHUNK_SIZE;
                b = GET_HEAP_PTR(p_al-1) + i * CHUNK_SIZE;
                p_al -= 2;
                if (av_result) {
                    res = av_stack[p_av];
                    p_av++;
                } else {
                    res = as_stack[p_as];
                    p_as++;
                }
                break;
            case  79: // 1001111 a-av b-as a-heap b-heap r-heap
                SET_RES_HEAP;
                a = GET_HEAP_PTR(p_al-2) + 3*i * CHUNK_SIZE;
                b = GET_HEAP_PTR(p_al-1) + i * CHUNK_SIZE;
                p_al -= 2;
                break;
                /* case  80: // 1010000 */
                /* case  81: // 1010001 */
                /* case  82: // 1010010 */
                /* case  83: // 1010011 */
                /* case  84: // 1010100 */
                /* case  85: // 1010101 */
                /* case  86: // 1010110 */
                /* case  87: // 1010111 */
                /* case  88: // 1011000 */
                /* case  89: // 1011001 */
                /* case  90: // 1011010 */
                /* case  91: // 1011011 */
                /* case  92: // 1011100 */
                /* case  93: // 1011101 */
                /* case  94: // 1011110 */
                /* case  95: // 1011111 */
                /*     INVALID; */
            case  96: // 1100000 a-av b-av a-stack b-stack r-stack
                a = av_stack[p_av-2];
                b = av_stack[p_av-1];
                if (av_result) {
                    res = av_stack[p_av-2];
                    p_av--;
                } else {
                    res = as_stack[p_as];
                    p_as++;
                    p_av -= 2;
                }
                break;
            case  97: // 1100001 a-av b-av a-stack b-stack r-stack
                SET_RES_HEAP;
                a = av_stack[p_av-2];
                b = av_stack[p_av-1];
                p_av -= 2;
                break;
            case  98: // 1100010 a-av b-av a-stack b-heap r-stack
                b = GET_HEAP_PTR(p_al-1) + 3*i * CHUNK_SIZE;
                p_al--;
                a = av_stack[p_av-1];
                if (av_result) {
                    res = av_stack[p_av-1];
                } else {
                    res = as_stack[p_as];
                    p_as++;
                    p_av--;
                }
                break;
            case  99: // 1100011 a-av b-av a-stack b-heap r-heap
                SET_RES_HEAP;
                a = av_stack[p_av-1];
                p_av--;
                b = GET_HEAP_PTR(p_al-1) + 3*i * CHUNK_SIZE;
                p_al--;
                break;
            case 100: // 1100100 a-av b-av a-heap b-stack r-stack
                a = GET_HEAP_PTR(p_al-1) + 3*i * CHUNK_SIZE;
                p_al -= 1;
                b  = av_stack[p_av-1];
                if (av_result) {
                    res = av_stack[p_av-1];
                } else {
                    res = as_stack[p_as];
                    p_av--;
                    p_as++;
                }
                break;
            case 101: // 1100101 a-av b-av a-heap b-stack r-heap
                SET_RES_HEAP;
                a = GET_HEAP_PTR(p_al-1) + 3*i * CHUNK_SIZE;
                p_al -= 1;
                b  = av_stack[p_av-1];
                if (av_result) {
                    res = av_stack[p_av-1];
                } else {
                    res = as_stack[p_as];
                    p_av--;
                    p_as++;
                }
                break;
            case 102: // 1100110 a-av b-av a-heap b-heap r-stack
                a = GET_HEAP_PTR(p_al-2) + 3*i * CHUNK_SIZE;
                b = GET_HEAP_PTR(p_al-1) + 3*i * CHUNK_SIZE;
                p_al -= 2;
                if (av_result) {
                    res = av_stack[p_av];
                    p_av++;
                } else {
                    res = as_stack[p_as];
                    p_as++;
                }
                break;
            case 103: // 1100111 a-av b-av a-heap b-heap r-heap
                SET_RES_HEAP;
                a = GET_HEAP_PTR(p_al-2) + 3*i * CHUNK_SIZE;
                b = GET_HEAP_PTR(p_al-1) + 3*i * CHUNK_SIZE;
                p_al -= 2;
                break;
            default:
                INVALID;
            }

            if (p_al >= STACK_DEPTH ||
                p_as >= STACK_DEPTH) {
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

    if (!(p_al == 0) || !(p_as == 0)) {
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
