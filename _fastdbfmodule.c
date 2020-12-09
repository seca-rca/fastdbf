#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API 1

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_common.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef _WIN32
#include <icu.h>
#else
#include <unicode/ucnv.h>
#include <unicode/ustring.h>
#endif

static PyObject *
fastdbf_load_i(PyObject *self, PyObject *args)
{
    PyObject *src_mview;
    PyArrayObject *dest_arr;
    long field_length;
    long recordlen;
    long numrecords;
    long offset;
    char *codepage;
    long i;

    if (!PyArg_ParseTuple(args, "OOlllls", &src_mview, &dest_arr, &field_length, &recordlen, &numrecords, &offset, &codepage)) {
        return NULL;
    }

    if (!PyArray_ISCARRAY(dest_arr)) {
        PyErr_SetString(PyExc_Exception, "numpy array must have flags: C_CONTIGUOUS, BEHAVED.");
        return NULL;
    }

    Py_buffer *buf = PyMemoryView_GET_BUFFER(src_mview);

    if (buf == NULL) {
        PyErr_SetString(PyExc_Exception, "Could not get memoryview buffer");
        return NULL;
    }

    char *src_buf = (char *)buf->buf;
    char *dest_buf = (char *)PyArray_DATA(dest_arr);

    for (i = 0; i < numrecords; i++) {
        memcpy(dest_buf + i * 4, src_buf + offset + i * recordlen, 4);
    }

    return Py_BuildValue("");
}

static PyObject *
fastdbf_load_l(PyObject *self, PyObject *args) {
    PyObject *src_mview;
    PyArrayObject *dest_arr;
    long field_length;
    long recordlen;
    long numrecords;
    long offset;
    char *codepage;
    long i;

    if (!PyArg_ParseTuple(args, "OOlllls", &src_mview, &dest_arr, &field_length, &recordlen, &numrecords, &offset, &codepage)) {
        return NULL;
    }

    if (!PyArray_ISCARRAY(dest_arr)) {
        PyErr_SetString(PyExc_Exception, "numpy array must have flags: C_CONTIGUOUS, BEHAVED.");
        return NULL;
    }

    Py_buffer *buf = PyMemoryView_GET_BUFFER(src_mview);

    if (buf == NULL) {
        PyErr_SetString(PyExc_Exception, "Could not get memoryview buffer");
        return NULL;
    }

    char *src_buf = (char *)buf->buf;
    char *dest_buf = (char *)PyArray_DATA(dest_arr);

    memset(dest_buf, 0, numrecords);

    for (i = 0; i < numrecords; i++) {
        if (src_buf[offset + i * recordlen] == 'T') {
            dest_buf[i] = 1;
        }
    }

    return Py_BuildValue("");
}

static PyObject *
fastdbf_load_n(PyObject *self, PyObject *args) {
    PyObject *src_mview;
    PyArrayObject *dest_arr;
    long field_length;
    long recordlen;
    long numrecords;
    long offset;
    char *codepage;
    long i;

    if (!PyArg_ParseTuple(args, "OOlllls", &src_mview, &dest_arr, &field_length, &recordlen, &numrecords, &offset, &codepage)) {
        return NULL;
    }

    if (!PyArray_ISCARRAY(dest_arr)) {
        PyErr_SetString(PyExc_Exception, "numpy array must have flags: C_CONTIGUOUS, BEHAVED.");
        return NULL;
    }

    if (field_length < 1 || field_length > 255) {
        PyErr_SetString(PyExc_Exception, "Field length out of range 1-255");
        return NULL;
    }

    Py_buffer *buf = PyMemoryView_GET_BUFFER(src_mview);

    if (buf == NULL) {
        PyErr_SetString(PyExc_Exception, "Could not get memoryview buffer");
        return NULL;
    }

    char *src_buf = (char *)buf->buf;
    float *dest_buf = (float *)PyArray_DATA(dest_arr);
    char *rec_buf = (char *)malloc(field_length + 1);

    if (rec_buf == NULL) {
        return PyErr_NoMemory();
    }

    float rec;

    for (i = 0; i < numrecords; i++) {
        memcpy(rec_buf, src_buf + offset + i * recordlen, field_length);
        rec_buf[field_length] = '\0';

        rec = strtof(rec_buf, NULL);

        if (errno == ERANGE) {
            dest_buf[i] = NPY_NANF;
            errno = 0;
        } else {
            dest_buf[i] = rec;
        }
    }

    free(rec_buf);

    return Py_BuildValue("");
}

#pragma region
// Taken from: https://github.com/numpy/numpy/blob/a9a44e9ac554a7b58804764e9149f38732baccb6/numpy/core/src/multiarray/datetime.c
// -------- START --------
//
/* Days per month, regular year and leap year */
NPY_NO_EXPORT int _days_per_month_table[2][12] = {
    { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
    { 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
};

/*
 * Returns 1 if the given year is a leap year, 0 otherwise.
 */
NPY_NO_EXPORT int
is_leapyear(npy_int64 year)
{
    return (year & 0x3) == 0 && /* year % 4 == 0 */
           ((year % 100) != 0 ||
            (year % 400) == 0);
}

/*
 * Calculates the days offset from the 1970 epoch.
 */
NPY_NO_EXPORT npy_int64
get_datetimestruct_days(const npy_datetimestruct *dts)
{
    int i, month;
    npy_int64 year, days = 0;
    int *month_lengths;

    year = dts->year - 1970;
    days = year * 365;

    /* Adjust for leap years */
    if (days >= 0) {
        /*
         * 1968 is the closest leap year before 1970.
         * Exclude the current year, so add 1.
         */
        year += 1;
        /* Add one day for each 4 years */
        days += year / 4;
        /* 1900 is the closest previous year divisible by 100 */
        year += 68;
        /* Subtract one day for each 100 years */
        days -= year / 100;
        /* 1600 is the closest previous year divisible by 400 */
        year += 300;
        /* Add one day for each 400 years */
        days += year / 400;
    }
    else {
        /*
         * 1972 is the closest later year after 1970.
         * Include the current year, so subtract 2.
         */
        year -= 2;
        /* Subtract one day for each 4 years */
        days += year / 4;
        /* 2000 is the closest later year divisible by 100 */
        year -= 28;
        /* Add one day for each 100 years */
        days -= year / 100;
        /* 2000 is also the closest later year divisible by 400 */
        /* Subtract one day for each 400 years */
        days += year / 400;
    }

    month_lengths = _days_per_month_table[is_leapyear(dts->year)];
    month = dts->month - 1;

    /* Add the months */
    for (i = 0; i < month; ++i) {
        days += month_lengths[i];
    }

    /* Add the days */
    days += dts->day - 1;

    return days;
}

/*
 * Converts a datetime from a datetimestruct to a datetime based
 * on some metadata. The date is assumed to be valid.
 *
 * TODO: If meta->num is really big, there could be overflow
 *
 * Returns 0 on success, -1 on failure.
 */
int
convert_datetimestruct_to_datetime(PyArray_DatetimeMetaData *meta,
                                    const npy_datetimestruct *dts,
                                    npy_datetime *out)
{
    npy_datetime ret;
    NPY_DATETIMEUNIT base = meta->base;

    /* If the datetimestruct is NaT, return NaT */
    if (dts->year == NPY_DATETIME_NAT) {
        *out = NPY_DATETIME_NAT;
        return 0;
    }

    /* Cannot instantiate a datetime with generic units */
    if (meta->base == NPY_FR_GENERIC) {
        PyErr_SetString(PyExc_ValueError,
                    "Cannot create a NumPy datetime other than NaT "
                    "with generic units");
        return -1;
    }

    if (base == NPY_FR_Y) {
        /* Truncate to the year */
        ret = dts->year - 1970;
    }
    else if (base == NPY_FR_M) {
        /* Truncate to the month */
        ret = 12 * (dts->year - 1970) + (dts->month - 1);
    }
    else {
        /* Otherwise calculate the number of days to start */
        npy_int64 days = get_datetimestruct_days(dts);

        switch (base) {
            case NPY_FR_W:
                /* Truncate to weeks */
                if (days >= 0) {
                    ret = days / 7;
                }
                else {
                    ret = (days - 6) / 7;
                }
                break;
            case NPY_FR_D:
                ret = days;
                break;
            case NPY_FR_h:
                ret = days * 24 +
                      dts->hour;
                break;
            case NPY_FR_m:
                ret = (days * 24 +
                      dts->hour) * 60 +
                      dts->min;
                break;
            case NPY_FR_s:
                ret = ((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec;
                break;
            case NPY_FR_ms:
                ret = (((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000 +
                      dts->us / 1000;
                break;
            case NPY_FR_us:
                ret = (((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us;
                break;
            case NPY_FR_ns:
                ret = ((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000 +
                      dts->ps / 1000;
                break;
            case NPY_FR_ps:
                ret = ((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000000 +
                      dts->ps;
                break;
            case NPY_FR_fs:
                /* only 2.6 hours */
                ret = (((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000000 +
                      dts->ps) * 1000 +
                      dts->as / 1000;
                break;
            case NPY_FR_as:
                /* only 9.2 secs */
                ret = (((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000000 +
                      dts->ps) * 1000000 +
                      dts->as;
                break;
            default:
                /* Something got corrupted */
                PyErr_SetString(PyExc_ValueError,
                        "NumPy datetime metadata with corrupt unit value");
                return -1;
        }
    }

    /* Divide by the multiplier */
    if (meta->num > 1) {
        if (ret >= 0) {
            ret /= meta->num;
        }
        else {
            ret = (ret - meta->num + 1) / meta->num;
        }
    }

    *out = ret;

    return 0;
}
//
// -------- END --------
#pragma endregion

static PyObject *
fastdbf_load_d(PyObject *self, PyObject *args) {
    PyObject *src_mview;
    PyArrayObject *dest_arr;
    long field_length;
    long recordlen;
    long numrecords;
    long offset;
    char *codepage;
    long i;

    if (!PyArg_ParseTuple(args, "OOlllls", &src_mview, &dest_arr, &field_length, &recordlen, &numrecords, &offset, &codepage)) {
        return NULL;
    }

    if (!PyArray_ISCARRAY(dest_arr)) {
        PyErr_SetString(PyExc_Exception, "numpy array must have flags: C_CONTIGUOUS, BEHAVED.");
        return NULL;
    }

    Py_buffer *buf = PyMemoryView_GET_BUFFER(src_mview);

    if (buf == NULL) {
        PyErr_SetString(PyExc_Exception, "Could not get memoryview buffer");
        return NULL;
    }

    char *src_buf = (char *)buf->buf;
    npy_datetime *dest_buf = (npy_datetime *)PyArray_DATA(dest_arr);

    npy_datetimestruct dts;
    dts.as = 0;
    dts.hour = 0;
    dts.min = 0;
    dts.ps = 0;
    dts.sec = 0;
    dts.us = 0;

    PyArray_DatetimeMetaData meta;
    meta.base = NPY_FR_D;
    meta.num = 1;

    npy_datetime ret;

    char year_buf[5];
    char month_buf[3];
    char day_buf[3];

    for (i = 0; i < numrecords; i++) {
        memcpy(year_buf, src_buf + offset + i * recordlen + 0, 4);
        memcpy(month_buf, src_buf + offset + i * recordlen + 4, 2);
        memcpy(day_buf, src_buf + offset + i * recordlen + 6, 2);

        year_buf[4] = '\0';
        month_buf[2] = '\0';
        day_buf[2] = '\0';

        dts.year = strtol(year_buf, NULL, 10);
        dts.month = strtol(month_buf, NULL, 10);
        dts.day = strtol(day_buf, NULL, 10);

        if (errno == ERANGE) {
            dest_buf[i] = NPY_DATETIME_NAT;
            errno = 0;
        } else {
            if (convert_datetimestruct_to_datetime(&meta, &dts, &ret)) {
                dest_buf[i] = NPY_DATETIME_NAT;
            } else {
                dest_buf[i] = ret;
            }
            
        }
    }

    return Py_BuildValue("");
}

static PyObject *
fastdbf_load_c(PyObject *self, PyObject *args) {
    PyObject *src_mview;
    PyArrayObject *dest_arr;
    long field_length;
    long recordlen;
    long numrecords;
    long offset;
    char *codepage;
    long i;

    if (!PyArg_ParseTuple(args, "OOlllls", &src_mview, &dest_arr, &field_length, &recordlen, &numrecords, &offset, &codepage)) {
        return NULL;
    }

    if (!PyArray_ISCARRAY(dest_arr)) {
        PyErr_SetString(PyExc_Exception, "numpy array must have flags: C_CONTIGUOUS, BEHAVED.");
        return NULL;
    }

    if (field_length < 1 || field_length > 255) {
        PyErr_SetString(PyExc_Exception, "Field length out of range 1-255");
        return NULL;
    }

    Py_buffer *buf = PyMemoryView_GET_BUFFER(src_mview);

    if (buf == NULL) {
        PyErr_SetString(PyExc_Exception, "Could not get memoryview buffer");
        return NULL;
    }

    char *src_buf = (char *)buf->buf;
    UChar32 *dest_buf = (UChar32 *)PyArray_DATA(dest_arr);
    npy_intp buf_size = PyArray_ITEMSIZE(dest_arr);
    UChar *u16_buf = (UChar *)malloc(2 * field_length * sizeof(UChar));
    UErrorCode err = U_ZERO_ERROR;

    if (u16_buf == NULL) {
        return PyErr_NoMemory();
    }

    UConverter *conv = ucnv_open(codepage, &err);

    if (conv == NULL) {
        return PyErr_Format(PyExc_Exception, "Could not create codec: %s", u_errorName(err));
    }

    int32_t len;

    for (i = 0; i < numrecords; i++) {
        ucnv_reset(conv);
        memset(u16_buf, 0, 2 * field_length * sizeof(UChar));

        len = ucnv_toUChars(conv, u16_buf, 2 * field_length, src_buf + offset + i * recordlen, field_length, &err);

        if (len > 0) {
            u_strToUTF32(&dest_buf[i * field_length], field_length, NULL, u16_buf, len, &err);
        } else {
            memset(dest_buf + i * buf_size, 0, buf_size);
        }

        err = U_ZERO_ERROR;
    }

    ucnv_close(conv);
    free(u16_buf);

    return Py_BuildValue("");
}

static PyMethodDef FastdbfMethods[] = {
    {"load_c",  fastdbf_load_c, METH_VARARGS, "load_c"},
    {"load_d",  fastdbf_load_d, METH_VARARGS, "load_d"},
    {"load_n",  fastdbf_load_n, METH_VARARGS, "load_n"},
    {"load_l",  fastdbf_load_l, METH_VARARGS, "load_l"},
    {"load_i",  fastdbf_load_i, METH_VARARGS, "load_i"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef fastdbfmodule = {
    PyModuleDef_HEAD_INIT,
    "_fastdbf",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    FastdbfMethods
};

PyMODINIT_FUNC
PyInit__fastdbf(void)
{
    PyObject *m = NULL;
    
    m = PyModule_Create(&fastdbfmodule);
    import_array();

    return m;
}