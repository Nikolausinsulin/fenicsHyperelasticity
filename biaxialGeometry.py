import gmsh
from dolfinx import mesh
import ufl
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI


def create_biaxial_geometry(h: float) -> tuple[mesh.Mesh, ufl.Measure]:

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("geometry_test")

    # Values as shown on schematic
    a = 17.5
    b = 17
    c = 17
    d = 23 / 2

    baseWidth = a + b
    stemHeight = c + d

    stemThickness = a
    baseThickness = d

    depth = 4  # extrusion depth

    assert stemThickness < baseWidth
    assert baseThickness < stemHeight

    p1 = gmsh.model.geo.addPoint(0, 0, 0, h)
    p2 = gmsh.model.geo.addPoint(0.5 * stemThickness, 0, 0, h)
    p3 = gmsh.model.geo.addPoint(baseWidth, 0, 0, h)
    p4 = gmsh.model.geo.addPoint(baseWidth, baseThickness, 0, h)
    p5 = gmsh.model.geo.addPoint(stemThickness, baseThickness, 0, h)
    p6 = gmsh.model.geo.addPoint(stemThickness, stemHeight, 0, h)
    p7 = gmsh.model.geo.addPoint(0, stemHeight, 0, h)
    p8 = gmsh.model.geo.addPoint(0, 0.5 * baseThickness, 0, h)
    p9 = gmsh.model.geo.addPoint(0.5 * stemThickness, 0.5 * baseThickness, 0, h)

    l_p1p2 = gmsh.model.geo.addLine(p1, p2)
    l_p2p3 = gmsh.model.geo.addLine(p2, p3)
    l_p3p4 = gmsh.model.geo.addLine(p3, p4)
    l_p4p5 = gmsh.model.geo.addLine(p4, p5)
    l_p5p6 = gmsh.model.geo.addLine(p5, p6)
    l_p6p7 = gmsh.model.geo.addLine(p6, p7)
    l_p7p8 = gmsh.model.geo.addLine(p7, p8)
    l_p8p9 = gmsh.model.geo.addLine(p8, p9)
    l_p9p2 = gmsh.model.geo.addLine(p9, p2)
    l_p8p1 = gmsh.model.geo.addLine(p8, p1)

    loop1 = gmsh.model.geo.addCurveLoop(
        [l_p2p3, l_p3p4, l_p4p5, l_p5p6, l_p6p7, l_p7p8, l_p8p9, l_p9p2]
    )
    surf1 = gmsh.model.geo.addPlaneSurface([loop1])

    loop2 = gmsh.model.geo.addCurveLoop([l_p1p2, -l_p9p2, -l_p8p9, l_p8p1])
    surf2 = gmsh.model.geo.addPlaneSurface([loop2])

    gmsh.model.geo.synchronize()

    # Extrude
    vol1 = gmsh.model.geo.extrude([(2, surf1)], 0, 0, depth)
    vol2 = gmsh.model.geo.extrude([(2, surf2)], 0, 0, depth)

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(3, [vol1[1][1]], 11)
    gmsh.model.addPhysicalGroup(3, [vol2[1][1]], 12)

    vol1VerticalLegBackside = vol1[7][1]
    vol2VerticalLegBackside = vol2[5][1]
    gmsh.model.addPhysicalGroup(
        2, [vol1VerticalLegBackside, vol2VerticalLegBackside], 1
    )

    vol1HorizontalLegBottomSide = vol1[2][1]
    vol2HorizontalLegBottomSide = vol2[2][1]
    gmsh.model.addPhysicalGroup(
        2, [vol1HorizontalLegBottomSide, vol2HorizontalLegBottomSide], 2
    )

    vol1HorizontalLegFarSide = vol1[3][1]
    gmsh.model.addPhysicalGroup(2, [vol1HorizontalLegFarSide], 3)

    vol1VerticalLegFarSide = vol1[6][1]
    gmsh.model.addPhysicalGroup(2, [vol1VerticalLegFarSide], 4)

    vol1FrontalArea = vol1[0][1]
    vol2FrontalArea = vol2[0][1]
    gmsh.model.addPhysicalGroup(2, [vol1FrontalArea, vol2FrontalArea], 5)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

    # generate the mesh
    gmsh.model.mesh.generate(3)

    data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    msh = data.mesh
    facet_tags = data.facet_tags
    cell_tags = data.cell_tags

    msh.topology.create_connectivity(2, 3)
    msh.topology.create_connectivity(1, 2)
    msh.topology.create_connectivity(1, 3)

    gmsh.finalize()
    return msh, facet_tags, cell_tags
