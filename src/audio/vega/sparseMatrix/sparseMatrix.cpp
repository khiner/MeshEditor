#include "sparseMatrix.h"

SparseMatrixOutline::SparseMatrixOutline(int numRows_) : numRows(numRows_) {
    columnEntries.clear();
    columnEntries.resize(numRows);
}

SparseMatrixOutline::~SparseMatrixOutline() {
    for (int i = 0; i < numRows; i++) columnEntries[i].clear();
    columnEntries.clear();
}

void SparseMatrixOutline::AddEntry(int i, int j, double value) {
    auto pos = columnEntries[i].find(j);
    if (pos != columnEntries[i].end()) pos->second += value;
    else columnEntries[i].insert({j, value});
}

double SparseMatrixOutline::GetEntry(int i, int j) const {
    const auto pos = columnEntries[i].find(j);
    return pos != columnEntries[i].end() ? pos->second : 0;
}

SparseMatrix::SparseMatrix(SparseMatrixOutline *sparseMatrixOutline) {
    numRows = sparseMatrixOutline->GetNumRows();
    // compressed row storage
    rowLength = (int *)malloc(sizeof(int) * numRows);
    columnIndices = (int **)malloc(sizeof(int *) * numRows);
    columnEntries = (double **)malloc(sizeof(double *) * numRows);

    for (int i = 0; i < numRows; i++) {
        rowLength[i] = (int)(sparseMatrixOutline->columnEntries[i].size());
        columnIndices[i] = (int *)malloc(sizeof(int) * rowLength[i]);
        columnEntries[i] = (double *)malloc(sizeof(double) * rowLength[i]);

        int j = 0;
        int prev = -1;
        for (auto pos = sparseMatrixOutline->columnEntries[i].begin(); pos != sparseMatrixOutline->columnEntries[i].end(); pos++) {
            columnIndices[i][j] = pos->first;
            if (columnIndices[i][j] <= prev) printf("Warning: entries not sorted in a row in a sparse matrix.\n");
            prev = columnIndices[i][j];
            columnEntries[i][j] = pos->second;
            j++;
        }
    }
}

SparseMatrix::~SparseMatrix() {
    for (int i = 0; i < numRows; i++) {
        free(columnIndices[i]);
        free(columnEntries[i]);
    }
    free(rowLength);
    free(columnIndices);
    free(columnEntries);
}

void SparseMatrix::ResetToZero() {
    for (int i = 0; i < numRows; i++) memset(columnEntries[i], 0, sizeof(double) * rowLength[i]);
}

// seeks the element in column jDense (in row "row")
// if not found, returns -1
int SparseMatrix::GetInverseIndex(int row, int jDense) const {
    for (int j = 0; j < rowLength[row]; j++) {
        if (columnIndices[row][j] == jDense) return j;
    }
    return -1;
}
