#include "Python.h"
#include "stdio.h"

#include "ops.h"

static PyObject *
hello_world(PyObject *self, PyObject *args)
{
    return Py_BuildValue("s", "hello, world!");
}


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

    case SS_POW:
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


static PyObject *vm_eval(PyObject *self, PyObject *args)
{
    // input is a tuple (opcode)
    Py_buffer ops;
    Py_buffer literals;
    int n_opt=0;
    int *c_ops=0;
    double *c_literals = 0;
    int stack_pointer = 0; // points to available stack location
    double stack[32];
    int i;

    if (!PyArg_ParseTuple(args, "z*z*", &ops, &literals))
        return NULL;
    c_ops = (int *)ops.buf;
    c_literals = (double *)literals.buf;
    n_opt = ops.len/4;
    //printf("got %i ops \n", n_opt);
    for (i=0; i<n_opt; i++) {
        //printf("opcode: %i\n", c_ops[i]);
        if (c_ops[i] <= 0) {
            //printf("pushing %lf\n", c_literals[-c_ops[i]]);
            stack[stack_pointer] = c_literals[-c_ops[i]];
            stack_pointer++;
        } else {
            switch (c_ops[i]) {
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
                //printf("got %i\n", c_ops[i]);
                PyErr_SetString(PyExc_ValueError,
                                "unknown opcode");
                return NULL;

            }
        }
    }

    return PyFloat_FromDouble(stack[0]);
}

static PyObject *call_test(PyObject *self, PyObject *args)
{
    return PyFloat_FromDouble(1.0);
}


// Module functions table.

static PyMethodDef
module_functions[] = {
    { "hello_world", hello_world, METH_VARARGS, "Say hello." },
    { "eval", eval, METH_VARARGS, "Say hello." },
    { "vm_eval", vm_eval, METH_VARARGS, "Say hello." },
    { "call_test", call_test, METH_VARARGS, "Say hello." },
    { NULL }
};

// This function is called to initialize the module.

void
initaeev(void)
{
    Py_InitModule3("aeev", module_functions, "A minimal module.");
}
