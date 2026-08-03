import gmsh
from dolfinx import mesh
import ufl
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI


def create_uniaxial_geometry(h: float) -> tuple[mesh.Mesh, ufl.Measure]:

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("geometry_test")

    # points going counter clockwise around geometry, starting in lower left corner
    # points along bottom
    P1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h)
    P2 = gmsh.model.geo.addPoint(21.5, 0.0, 0.0, h)
    P3 = gmsh.model.geo.addPoint(45.0, 5.0, 0.0, h)
    P4 = gmsh.model.geo.addPoint(50.0, 5.0, 0.0, h)
    P5 = gmsh.model.geo.addPoint(100.0, 5.0, 0.0, h)
    P6 = gmsh.model.geo.addPoint(105.0, 5.0, 0.0, h)
    P7 = gmsh.model.geo.addPoint(128.5, 0, 0.0, h)
    P8 = gmsh.model.geo.addPoint(150.0, 0, 0.0, h)
    # points along top
    P9 = gmsh.model.geo.addPoint(150.0, 20.0, 0.0, h)
    P10 = gmsh.model.geo.addPoint(128.5, 20.0, 0.0, h)
    P11 = gmsh.model.geo.addPoint(105.0, 15.0, 0.0, h)
    P12 = gmsh.model.geo.addPoint(100.0, 15.0, 0.0, h)
    P13 = gmsh.model.geo.addPoint(50.0, 15.0, 0.0, h)
    P14 = gmsh.model.geo.addPoint(45.0, 15.0, 0.0, h)
    P15 = gmsh.model.geo.addPoint(21.5, 20.0, 0.0, h)
    P16 = gmsh.model.geo.addPoint(0.0, 20.0, 0.0, h)

    # center of circle
    P17 = gmsh.model.geo.addPoint(45.0, -55.0, 0.0, h)
    P18 = gmsh.model.geo.addPoint(105.0, -55.0, 0.0, h)
    P19 = gmsh.model.geo.addPoint(105.0, 75.0, 0.0, h)
    P20 = gmsh.model.geo.addPoint(45.0, 75.0, 0.0, h)

    # external line
    l1 = gmsh.model.geo.addLine(P1, P2)
    l2 = gmsh.model.geo.addCircleArc(P2, P17, P3)
    l3 = gmsh.model.geo.addLine(P3, P4)
    l4 = gmsh.model.geo.addLine(P4, P5)
    l5 = gmsh.model.geo.addLine(P5, P6)
    l6 = gmsh.model.geo.addCircleArc(P6, P18, P7)
    l7 = gmsh.model.geo.addLine(P7, P8)
    l8 = gmsh.model.geo.addLine(P8, P9)
    l9 = gmsh.model.geo.addLine(P9, P10)
    l10 = gmsh.model.geo.addCircleArc(P10, P19, P11)
    l11 = gmsh.model.geo.addLine(P11, P12)
    l12 = gmsh.model.geo.addLine(P12, P13)
    l13 = gmsh.model.geo.addLine(P13, P14)
    l14 = gmsh.model.geo.addCircleArc(P14, P20, P15)
    l15 = gmsh.model.geo.addLine(P15, P16)
    l16 = gmsh.model.geo.addLine(P16, P1)

    # Inside line to divide the geometry
    lcut1 = gmsh.model.geo.addLine(P2, P15)
    lcut2 = gmsh.model.geo.addLine(P4, P13)
    lcut3 = gmsh.model.geo.addLine(P5, P12)
    lcut4 = gmsh.model.geo.addLine(P7, P10)

    gmsh.model.geo.synchronize()

    # creation of 5 different faces
    # left
    loop_left = gmsh.model.geo.addCurveLoop([l1, lcut1, l15, l16])
    surface_left = gmsh.model.geo.addPlaneSurface([loop_left])

    # left_transition
    loop_left_transition = gmsh.model.geo.addCurveLoop(
        [l2, l3, lcut2, l13, l14, -lcut1]
    )
    surface_left_transition = gmsh.model.geo.addPlaneSurface([loop_left_transition])

    # center
    loop_center = gmsh.model.geo.addCurveLoop([l4, lcut3, l12, -lcut2])
    surface_center = gmsh.model.geo.addPlaneSurface([loop_center])

    # right transition
    loop_right_transition = gmsh.model.geo.addCurveLoop(
        [l5, l6, lcut4, l10, l11, -lcut3]
    )
    surface_right_transition = gmsh.model.geo.addPlaneSurface([loop_right_transition])

    # right
    loop_right = gmsh.model.geo.addCurveLoop([l7, l8, l9, -lcut4])
    surface_right = gmsh.model.geo.addPlaneSurface([loop_right])

    # extrusion
    extruded_left = gmsh.model.geo.extrude([(2, surface_left)], 0, 0, 4)
    extruded_left_transition = gmsh.model.geo.extrude(
        [(2, surface_left_transition)], 0, 0, 4
    )
    extruded_center = gmsh.model.geo.extrude([(2, surface_center)], 0, 0, 4)
    extruded_right_transition = gmsh.model.geo.extrude(
        [(2, surface_right_transition)], 0, 0, 4
    )
    extruded_right = gmsh.model.geo.extrude([(2, surface_right)], 0, 0, 4)

    gmsh.model.geo.synchronize()

    # set the size of the mesh
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

    # Create the volume tag
    gmsh.model.addPhysicalGroup(3, [extruded_left[1][1]], 1)
    gmsh.model.addPhysicalGroup(3, [extruded_left_transition[1][1]], 2)
    gmsh.model.addPhysicalGroup(3, [extruded_center[1][1]], 3)
    gmsh.model.addPhysicalGroup(3, [extruded_right_transition[1][1]], 4)
    gmsh.model.addPhysicalGroup(3, [extruded_right[1][1]], 5)

    # create face tag for the boundary
    # left
    top_surface_left = extruded_left[0][1]
    gmsh.model.addPhysicalGroup(2, [top_surface_left, surface_left], 1000)

    # right
    top_surface_right = extruded_right[0][1]
    gmsh.model.addPhysicalGroup(2, [top_surface_right, surface_right], 2000)

    gmsh.model.mesh.generate(3)

    data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    msh = data.mesh
    facet_tags = data.facet_tags

    msh.topology.create_connectivity(2, 3)
    msh.topology.create_connectivity(1, 2)
    msh.topology.create_connectivity(1, 3)

    gmsh.finalize()
    return msh, facet_tags
