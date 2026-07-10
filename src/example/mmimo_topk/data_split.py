"""
여러 개의 파일이 이어붙여진 형태의 입력 데이터를, 파일 단위로 train/test 로 분할하는 모듈.

원본 파일들이 각각 no_dataInFile 개의 샘플을 담고 있고, 그 파일들을 순서대로
이어붙인(concatenate) 하나의 배열이 주어졌을 때, 앞쪽 no_train_files 개 파일분은
train, 나머지는 test 로 슬라이싱한다.
(mMIMO-80 mf - FC HTH for Validation.py 의 "Assign Train and Test Data" 로직을 일반화)

원본 pickle 파일은 수 GB 단위로 크기 때문에, 전체를 메모리에 올리는 load_pickle
대신 실제 분할 작업(split_file_range / take_test_rows)은 pickle_memmap 모듈을
통해 필요한 행 구간만 memmap으로 읽는다.
"""

import pickle

from pickle_memmap import load_row_range, peek_ndarray_info


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def split_by_file_units(data, no_dataInFile, no_train_files):
    """
    data: shape (no_dataInFile * 전체 파일 수, ...) 인 배열
    no_dataInFile: 파일 하나에 들어있는 샘플 개수
    no_train_files: 앞에서부터 train 으로 사용할 파일 개수

    반환: (train, test)
    """
    no_totalData = no_dataInFile * no_train_files
    train = data[:no_totalData]
    test = data[no_totalData:]
    return train, test


def take_test_files(data, no_dataInFile, no_test_files):
    """
    data 뒤쪽 no_test_files 개 파일분만 test 로 추출한다. (train 부분은 만들지 않음)

    data: shape (no_dataInFile * 전체 파일 수, ...) 인 배열
    no_dataInFile: 파일 하나에 들어있는 샘플 개수
    no_test_files: 뒤에서부터 test 로 사용할 파일 개수

    반환: test
    """
    no_testData = no_dataInFile * no_test_files
    return data[-no_testData:]


def test_row_range(filepath, no_dataInFile, no_test_files):
    """
    pickle 파일 전체를 읽지 않고, peek_ndarray_info로 얻은 전체 행 수만으로
    test 구간의 절대 행 범위 [start, end) 를 계산한다.
    """
    info = peek_ndarray_info(filepath)
    total_rows = info.shape[0]
    no_testData = no_dataInFile * no_test_files
    return total_rows - no_testData, total_rows


def load_test_rows(filepath, no_dataInFile, no_test_files):
    """
    take_test_files와 동일한 결과를, 파일 전체를 메모리에 올리지 않고
    memmap으로 test 구간(뒤쪽 no_test_files개 파일분)만 읽어서 반환한다.
    """
    start, end = test_row_range(filepath, no_dataInFile, no_test_files)
    return load_row_range(filepath, start, end)


if __name__ == "__main__":
    # 사용 예시 (mMIMO-80 mf - FC HTH for Validation.py 기준)
    # 3GB대 원본 파일이므로 load_pickle(전체 로드) 대신 load_test_rows(memmap)를 사용한다.
    input_path = "C:\AI_Verification\wireless\Pickle/mMIMO_AS_training_data_20000_80_H_HTH_ORG_1D-003.pickle"

    test_input = load_test_rows(input_path, no_dataInFile=20000, no_test_files=2)

    print("test input:", test_input.shape)
