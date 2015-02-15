#include "Python.h"
#include "stdio.h"

static PyObject *
hello_world(PyObject *self, PyObject *args)
{
    return Py_BuildValue("s", "hello, world!");
}

#define SS_ADD  0
#define SS_SUB  1
#define SS_MUL  2
#define SS_DIV  3
#define S_NEGATE  4
#define S_POW  5
#define I_SCALAR  400

double eval_double(PyObject *cell)
{
    int op_code = PyInt_AS_LONG(PyTuple_GET_ITEM(cell,0));
    switch (op_code){

    case SS_ADD:
        return eval_double(PyTuple_GET_ITEM(cell, 1)) +
               eval_double(PyTuple_GET_ITEM(cell, 2));

    case SS_SUB:
        return eval_double(PyTuple_GET_ITEM(cell, 1)) -
               eval_double(PyTuple_GET_ITEM(cell, 2));

    case SS_MUL:
        return eval_double(PyTuple_GET_ITEM(cell, 1)) *
               eval_double(PyTuple_GET_ITEM(cell, 2));

    case SS_DIV:
        return eval_double(PyTuple_GET_ITEM(cell, 1)) /
               eval_double(PyTuple_GET_ITEM(cell, 2));

    case S_NEGATE:
        return -eval_double(PyTuple_GET_ITEM(cell, 1));

    case S_POW:
        return pow(eval_double(PyTuple_GET_ITEM(cell, 1)),
                   eval_double(PyTuple_GET_ITEM(cell, 2)));

    case I_SCALAR:
        return PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(cell, 1));

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
    return PyFloat_FromDouble(eval_double(cell));
}


// Module functions table.

static PyMethodDef
module_functions[] = {
    { "hello_world", hello_world, METH_VARARGS, "Say hello." },
    { "eval", eval, METH_VARARGS, "Say hello." },
    { NULL }
};

// This function is called to initialize the module.

void
initaeev(void)
{
    Py_InitModule3("aeev", module_functions, "A minimal module.");
}
