#pragma once

#include <map>
#include <vector>

struct SparseMatrix;

struct SparseMatrixOutline {
    SparseMatrixOutline(int numRows_) : numRows(numRows_) {
        columnEntries.clear();
        columnEntries.resize(numRows);
    }

    ~SparseMatrixOutline() {
        for (int i = 0; i < numRows; i++) columnEntries[i].clear();
        columnEntries.clear();
    }

    void AddEntry(int i, int j, double value) {
        auto pos = columnEntries[i].find(j);
        if (pos != columnEntries[i].end()) pos->second += value;
        else columnEntries[i].emplace(j, value);
    }
    double GetEntry(int i, int j) const {
        const auto pos = columnEntries[i].find(j);
        return pos != columnEntries[i].end() ? pos->second : 0;
    }

    const std::map<int, double> &GetRow(int i) const { return columnEntries[i]; }

    int numRows;
    std::vector<std::map<int, double>> columnEntries;
};

struct SparseMatrix {
    SparseMatrix(const SparseMatrixOutline &outline) : numRows(outline.numRows) {
        // compressed row storage
        rowLength = (int *)malloc(sizeof(int) * numRows);
        columnIndices = (int **)malloc(sizeof(int *) * numRows);
        columnEntries = (double **)malloc(sizeof(double *) * numRows);
        for (int i = 0; i < numRows; i++) {
            rowLength[i] = (int)(outline.columnEntries[i].size());
            columnIndices[i] = (int *)malloc(sizeof(int) * rowLength[i]);
            columnEntries[i] = (double *)malloc(sizeof(double) * rowLength[i]);

            int j = 0;
            int prev = -1;
            for (auto pos = outline.columnEntries[i].begin(); pos != outline.columnEntries[i].end(); pos++) {
                columnIndices[i][j] = pos->first;
                if (columnIndices[i][j] <= prev) printf("Warning: entries not sorted in a row in a sparse matrix.\n");
                prev = columnIndices[i][j];
                columnEntries[i][j] = pos->second;
                j++;
            }
        }
    }

    ~SparseMatrix() {
        for (int i = 0; i < numRows; i++) {
            free(columnIndices[i]);
            free(columnEntries[i]);
        }
        free(rowLength);
        free(columnIndices);
        free(columnEntries);
    }

    // Add value to the j-th sparse entry in the given row (NOT to matrix element at (row,j))
    void AddEntry(int row, int j, double value) { columnEntries[row][j] += value; }
    void ResetToZero() {
        for (int i = 0; i < numRows; i++) memset(columnEntries[i], 0, sizeof(double) * rowLength[i]);
    }

    int Getn() const { return numRows; } // get the number of rows
    int GetNumRows() const { return numRows; }
    int GetRowLength(int row) const { return rowLength[row]; }
    // Returns the j-th sparse entry in row i (NOT matrix element at (row, j))
    double GetEntry(int row, int j) const { return columnEntries[row][j]; }

    // Returns the column index of the j-th sparse entry in the given row
    int GetColumnIndex(int row, int j) const { return columnIndices[row][j]; }

    // Find the compressed column index of element at location (row, jDense)
    // returns -1 if column not found
    int GetInverseIndex(int row, int jDense) const {
        for (int j = 0; j < rowLength[row]; j++) {
            if (columnIndices[row][j] == jDense) return j;
        }
        return -1;
    }

    double **GetDataHandle() const { return columnEntries; }

private:
    // compressed row storage
    int numRows; // number of rows
    int *rowLength; // length of each row
    int **columnIndices; // indices of columns of non-zero entries in each row
    double **columnEntries; // values of non-zero entries in each row
};
