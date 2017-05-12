"""
@author lmiguelmh
@since 20170501
"""
import codecs
import pickle


def save(file_pkz, content, compress=True):
    print('saving "%s"' % file_pkz)
    if compress:
        compressed_content = codecs.encode(pickle.dumps(content), 'zlib_codec')
    else:
        compressed_content = pickle.dumps(content)
    with open(file_pkz, 'wb') as f:
        f.write(compressed_content)


def retrieve(file_pkz, decompress=True):
    print('retrieving "%s"' % file_pkz)
    with open(file_pkz, 'rb') as f:
        compressed_content = f.read()
    if decompress:
        return pickle.loads(codecs.decode(compressed_content, 'zlib_codec'))
    else:
        return pickle.loads(compressed_content)
