#include "SceneDefaults.h"

#include <glm/trigonometric.hpp>
SceneDefaults::SceneDefaults()
    : World{.Origin{0, 0, 0}, .Up{0, 1, 0}},
      ViewCamera{
          {0, 0, 2},
          {0, 0, 0},
          {Perspective{.FieldOfViewRad = glm::radians(60.f), .FarClip = 100.f, .NearClip = 0.01f}},
      },
      Lights{
          .ViewColor = {1, 1, 1},
          .AmbientIntensity = 0.1f,
          .DirectionalColor = {1, 1, 1},
          .DirectionalIntensity = 0.15f,
          .Direction = {-1, -1, -1},
      },
      ViewportTheme{
          .Colors{
              .Wire{0, 0, 0},
              .WireEdit{0, 0, 0},
              .ObjectActive{1, 0.627f, 0.157f},
              .ObjectSelected{0.929f, 0.341f, 0},
              .Light{0, 0, 0, 0.314f},
              .Vertex{0, 0, 0},
              .VertexSelected{1, 0.478f, 0},
              .EdgeSelectedIncidental{1, 0.6f, 0},
              .EdgeSelected{1, 0.847f, 0},
              .FaceSelectedIncidental{1, 0.639f, 0, 0.2f},
              .FaceSelected{1, 0.718f, 0, 0.2f},
              .ElementActive{1, 1, 1, 0.2f},
              .FaceNormal{0.133f, 0.867f, 0.867f},
              .VertexNormal{0.137f, 0.380f, 0.867f},
              .Transform{1, 1, 1},
          },
          .SilhouetteEdgeWidth = 1,
      } {}
