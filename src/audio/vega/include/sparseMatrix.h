#pragma once

#include <map>
#include <vector>

struct SparseMatrix;

struct SparseMatrixOutline {
    // makes an empty sparse matrix with numRows rows
    SparseMatrixOutline(int numRows);
    ~SparseMatrixOutline();

    // add entry at location (i,j) in the matrix
    void AddEntry(int i, int j, double value = 0.0);

    int Getn() const { return numRows; } // get number of rows
    int GetNumRows() const { return numRows; } // get number of rows
    double GetEntry(int i, int j) const; // returns the matrix entry at location (i,j) in the matrix (or zero if entry has not been assigned)

    const std::map<int, double> &GetRow(int i) const { return columnEntries[i]; }

    int numRows;
    std::vector<std::map<int, double>> columnEntries;
};

struct SparseMatrix {
    SparseMatrix(SparseMatrixOutline *sparseMatrixOutline); // create it from the outline
    ~SparseMatrix();

    // add value to the j-th sparse entry in the given row (NOT to matrix element at (row,j))
    void AddEntry(int row, int j, double value) { columnEntries[row][j] += value; }
    void ResetToZero(); // reset all entries to zero

    int Getn() const { return numRows; } // get the number of rows
    int GetNumRows() const { return numRows; }
    int GetRowLength(int row) const { return rowLength[row]; }
    // returns the j-th sparse entry in row i (NOT matrix element at (row, j))
    double GetEntry(int row, int j) const { return columnEntries[row][j]; }
    // returns the column index of the j-th sparse entry in the given row
    int GetColumnIndex(int row, int j) const { return columnIndices[row][j]; }

    // finds the compressed column index of element at location (row, jDense)
    // returns -1 if column not found
    int GetInverseIndex(int row, int jDense) const;

    double **GetDataHandle() const { return columnEntries; }

private:
    // compressed row storage
    int numRows; // number of rows
    int *rowLength; // length of each row
    int **columnIndices; // indices of columns of non-zero entries in each row
    double **columnEntries; // values of non-zero entries in each row
};
