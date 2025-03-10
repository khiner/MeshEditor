#include "generateMassMatrix.h"

void GenerateMassMatrix::computeMassMatrix(TetMesh *tetMesh, SparseMatrix **massMatrix) {
    int n = tetMesh->getNumVertices();
    int numElementVertices = tetMesh->getNumElementVertices();
    double *buffer = (double *)malloc(sizeof(double) * numElementVertices * numElementVertices);

    SparseMatrixOutline *massMatrixOutline = new SparseMatrixOutline(3 * n);
    for (int el = 0; el < tetMesh->getNumElements(); el++) {
      tetMesh->computeElementMassMatrix(el, buffer);
        for (int i = 0; i < numElementVertices; i++)
            for (int j = 0; j < numElementVertices; j++) {
                double entry = buffer[numElementVertices * j + i];
                int indexi = tetMesh->getVertexIndex(el, i);
                int indexj = tetMesh->getVertexIndex(el, j);
                massMatrixOutline->AddEntry(3 * indexi + 0, 3 * indexj + 0, entry);
                massMatrixOutline->AddEntry(3 * indexi + 1, 3 * indexj + 1, entry);
                massMatrixOutline->AddEntry(3 * indexi + 2, 3 * indexj + 2, entry);
            }
    }

    (*massMatrix) = new SparseMatrix(massMatrixOutline);
    delete (massMatrixOutline);

    free(buffer);
}
