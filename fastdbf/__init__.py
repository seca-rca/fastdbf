# DBF file format information sources:
#
# https://www.dbf2002.com/dbf-file-format.html
#
# https://www.dbase.com/Knowledgebase/INT/db7_file_fmt.htm
#
# http://www.independent-software.com/dbase-dbf-dbt-file-format.html

import pandas as pd
import numpy as np
import datetime
import mmap
import struct

HAVE_C_EXT = None

try:
    import _fastdbf as fd
    HAVE_C_EXT = True
except ImportError:
    HAVE_C_EXT = False

    def _parse_cp(data, codepage):
        return data.decode(codepage)

    def _parse_d(data):
        try:
            str_ = data.decode('ascii')
            return np.datetime64(f"{str_[:4]}-{str_[4:6]}-{str_[6:8]}", "D")
        except ValueError:
            return None

    def _parse_n(data):
        try:
            return np.float32(data)
        except ValueError:
            return None

    def _parse_i(data):
        return np.frombuffer(data, dtype='<i4')

    def _parse_l(data):
        if data == b'T':
            return True
        elif data == b'F':
            return False
        
        return None

codepages = {
    0x00: 'ascii',
    0x01: 'cp437',
    0x02: 'cp850',
    0x03: 'cp1252',
    0x04: 'mac_roman',
    0x08: 'cp865',
    0x09: 'cp437',
    0x0A: 'cp850',
    0x0B: 'cp437',
    0x0D: 'cp437',
    0x0E: 'cp850',
    0x0F: 'cp437',
    0x10: 'cp850',
    0x11: 'cp437',
    0x12: 'cp850',
    0x13: 'cp932',
    0x14: 'cp850',
    0x15: 'cp437',
    0x16: 'cp850',
    0x17: 'cp865',
    0x18: 'cp437',
    0x19: 'cp437',
    0x1A: 'cp850',
    0x1B: 'cp437',
    0x1C: 'cp863',
    0x1D: 'cp850',
    0x1F: 'cp852',
    0x22: 'cp852',
    0x23: 'cp852',
    0x24: 'cp860',
    0x25: 'cp850',
    0x26: 'cp866',
    0x37: 'cp850',
    0x40: 'cp852',
    0x4D: 'cp936',
    0x4E: 'cp949',
    0x4F: 'cp950',
    0x50: 'cp874',
    0x57: 'cp1252',
    0x58: 'cp1252',
    0x59: 'cp1252',
    0x64: 'cp852',
    0x65: 'cp866',
    0x66: 'cp865',
    0x67: 'cp861',
    0x6a: 'cp737',
    0x6b: 'cp857',
    0x78: 'cp950',
    0x79: 'cp949',
    0x7a: 'cp936',
    0x7b: 'cp932',
    0x7c: 'cp874',
    0x7d: 'cp1255',
    0x7e: 'cp1256',
    0xc8: 'cp1250',
    0xc9: 'cp1251',
    0xca: 'cp1254',
    0xcb: 'cp1253',
    0x96: 'mac_cyrillic',
    0x97: 'mac_latin2',
    0x98: 'mac_greek'
}

def to_df(file_name, keep=None, lowernames=True):
    header_values = struct.Struct('<BBBBLHHHBBLLLBBH')
    header_keys = [
        'dbversion',
        'year',
        'month',
        'day',
        'numrecords',
        'headerlen',
        'recordlen',
        'reserved1',
        'incomplete_transaction',
        'encryption_flag',
        'free_record_thread',
        'reserved2',
        'reserved3',
        'mdx_flag',
        'language_driver',
        'reserved4'
    ]

    field_values = struct.Struct('<11scLBBHBBBB7sB')
    field_keys = ['name',
        'type',
        'address',
        'length',
        'decimal_count',
        'reserved1',
        'workarea_id',
        'reserved2',
        'reserved3',
        'set_fields_flag',
        'reserved4',
        'index_field_flag'
    ]

    with open(file_name, "rb") as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        offset = 0

        header = dict(zip(header_keys, header_values.unpack_from(m, offset)))
        header['codepage'] = codepages[header['language_driver']]

        if HAVE_C_EXT:
            loader = {
                'C': fd.load_c,
                'D': fd.load_d,
                'N': fd.load_n,
                'L': fd.load_l,
                'I': fd.load_i
            }

            dtypes = {
                'C': lambda len: f"U{len:d}", # UTF-32 native
                'D': lambda len: 'M8[D]', # converted from string, native
                'N': lambda len: 'f4', # strtof, native
                'L': lambda len: '?', # stored as 1-byte, converted from string
                'I': lambda len: '=i4' # memcpy, defined as little endian, see http://www.independent-software.com/dbase-dbf-dbt-file-format.html
            }
        else:
            def _parse_c(data):
                return _parse_cp(data, header['codepage'])

            parser = {
                'C': _parse_c,
                'D': _parse_d,
                'N': _parse_n,
                'L': _parse_l,
                'I': _parse_i
            }

            dtypes = {
                'C': lambda len: f"U{len:d}",
                'D': lambda len: 'M8[D]',
                'N': lambda len: 'f4',
                'L': lambda len: '?',
                'I': lambda len: 'i4'
            }

        supported_types = dtypes.keys()

        fields = []
        offset = header_values.size

        while True:
            m.seek(offset)
            eol = m.read(1)
            if eol in (b'\r'):
                break

            field = dict(zip(field_keys, field_values.unpack_from(m, offset)))

            field['type'] = chr(ord(field['type']))
            field['name'] = str(field['name'].split(b'\0')[0], header['codepage']).lower()

            if ((not keep) or (field['name'].lower() in keep)) and field['type'] in supported_types:
                fields.append(field)
            
            offset += field_values.size

        columns = {}
        recordlen = header['recordlen']
        numrecords = header['numrecords']

        for field in fields:
            offset = header['headerlen'] + field['address']
            
            field_length = field['length']
            field_type = field['type']
            field_name = field['name']

            if HAVE_C_EXT:
                loader_func = loader[field_type]
                arr = np.require(np.zeros(numrecords, dtype=dtypes[field_type](field_length)), requirements='CAW') # C_CONTIGUOUS, ALIGNED, WRITEABLE

                loader_func(memoryview(m), arr, field_length, recordlen, numrecords, offset, header['codepage'])
            else:
                parser_func = parser[field_type]
                arr = np.empty(numrecords, dtype=dtypes[field_type](field_length))

                for i in range(numrecords):
                    m.seek(offset + i * recordlen)
                    arr[i] = parser_func(m.read(field_length))
            
            if field_type == 'D':
                columns[field_name] = pd.to_datetime(arr, errors='coerce')
            elif field_type == 'C':
                columns[field_name] = np.char.rstrip(arr, "\0 ")
            else:
                columns[field_name] = arr
        
        return pd.DataFrame.from_dict(columns)

import fnmatch
import re
import os
import concurrent.futures

# Based on https://gist.github.com/techtonik/5694830
def findfile(pattern, dir_='.'):
    rule = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
    matches = [name for name in os.listdir(dir_) if rule.match(name)]

    if len(matches) > 0:
        return os.path.realpath(os.path.join(dir_, matches[0]))

def multi_df(dirs, filename):
    fnames = [f for f in [findfile(filename, dir_) for dir_ in dirs] if f]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        return pd.concat(executor.map(to_df, fnames), ignore_index=True, axis=0, copy=False)