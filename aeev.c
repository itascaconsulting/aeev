#include "Python.h"
#include "numpy/noprefix.h"
#include "stdio.h"
#include "ops.h"

static PyObject *
hello_world(PyObject *self, PyObject *args)
{
    return Py_BuildValue("s", "hello, world!");
}

double eval_double(PyObject *cell, int index)
{
    int op_code = PyInt_AS_LONG(PyTuple_GET_ITEM(cell,0));
    switch (op_code){

    case SS_ADD:
        return eval_double(PyTuple_GET_ITEM(cell, 1), index) +
               eval_double(PyTuple_GET_ITEM(cell, 2), index);

    case SS_SUB:
        return eval_double(PyTuple_GET_ITEM(cell, 1), index) -
               eval_double(PyTuple_GET_ITEM(cell, 2), index);

    case SS_MUL:
        return eval_double(PyTuple_GET_ITEM(cell, 1), index) *
               eval_double(PyTuple_GET_ITEM(cell, 2), index);

    case SS_DIV:
        return eval_double(PyTuple_GET_ITEM(cell, 1), index) /
               eval_double(PyTuple_GET_ITEM(cell, 2), index);

    case S_NEGATE:
        return -eval_double(PyTuple_GET_ITEM(cell, 1), index);

    case SS_POW:
        return pow(eval_double(PyTuple_GET_ITEM(cell, 1), index),
                   eval_double(PyTuple_GET_ITEM(cell, 2), index));

    case I_SCALAR:
        return PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(cell, 1));

    case IA_SCALAR:
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
        if (c_op <= 0) {
            stack[stack_pointer] = PyFloat_AS_DOUBLE(
                PyTuple_GET_ITEM(literals, -c_op));
            stack_pointer++;
        } else {
            switch (c_op) {
            case SS_ADD:
                stack[stack_pointer-2] = stack[stack_pointer-2] +
                                         stack[stack_pointer-1];
                stack_pointer--;
                break;
            case SS_POW:
                stack[stack_pointer-2] = pow(stack[stack_pointer-2],
                                             stack[stack_pointer-1]);
                stack_pointer--;
                break;
            default:
                PyErr_SetString(PyExc_ValueError,
                                "unknown opcode");
                return NULL;

            }
        }
    }
    return PyFloat_FromDouble(stack[0]);
}

#define CHUNK_SIZE 256

static PyObject *call_test_chunk(PyObject *self, PyObject *args)
{
    int size=0;
    int i=0;
    int j=0;
    PyArrayObject *ar0;
    PyArrayObject *ar1;
    PyArrayObject *ar2;

    double *data0;
    double *data1;
    double *data2;

    double local_a[CHUNK_SIZE];
    double local_b[CHUNK_SIZE];
    double local_c[CHUNK_SIZE];

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
    int nsteps = size/CHUNK_SIZE;

    for (i=0; i<nsteps; i++) {
        for (j=0; j<CHUNK_SIZE; j++) { local_a[j] = data0[i*CHUNK_SIZE + j]; }
        for (j=0; j<CHUNK_SIZE; j++) { local_b[j] = data1[i*CHUNK_SIZE + j]; }
        for (j=0; j<CHUNK_SIZE; j++) { local_c[j] = local_a[j] + local_b[j]; }
        for (j=0; j<CHUNK_SIZE; j++) { data2[i+CHUNK_SIZE + j] = local_c[j]; }
    }

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


// Module functions table.
static PyMethodDef
module_functions[] = {
    { "hello_world", hello_world, METH_VARARGS, "Say hello." },
    { "eval", eval, METH_VARARGS, "Say hello." },
    { "array_eval", array_eval, METH_VARARGS, "Say hello." },
    { "vm_eval", vm_eval, METH_VARARGS, "Say hello." },
    { "call_test", call_test, METH_VARARGS, "Say hello." },
    { "call_test_chunk", call_test_chunk, METH_VARARGS, "Say hello." },
    { NULL }
};

// This function is called to initialize the module.

void
initaeev(void)
{
    Py_InitModule3("aeev", module_functions, "A minimal module.");
    import_array();
}
