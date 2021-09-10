from panda3d.core import GeomVertexFormat, GeomVertexData, Geom, GeomTriangles, GeomVertexWriter, GeomNode, NodePath, LineSegs
from panda3d.core import LVector3
import math


def normalized(*args):
    myVec = LVector3(*args)
    myVec.normalize()
    return myVec


class VertexDataHelper():
    def __init__(self):
        self.format = GeomVertexFormat.getV3n3cpt2()
        self.vdata = GeomVertexData('square', self.format, Geom.UHDynamic)
        self.vertex = GeomVertexWriter(self.vdata, 'vertex')
        self.normal = GeomVertexWriter(self.vdata, 'normal')
        self.color = GeomVertexWriter(self.vdata, 'color')
        self.texcoord = GeomVertexWriter(self.vdata, 'texcoord')

        self.tris = GeomTriangles(Geom.UHDynamic)


class GeomPrimitives:
    @staticmethod
    def Cylinder(n):
        format = GeomVertexFormat.getV3n3cpt2()
        vdata = GeomVertexData('square', format, Geom.UHDynamic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        texcoord = GeomVertexWriter(vdata, 'texcoord')
        tris = GeomTriangles(Geom.UHDynamic)

        for i in range(n+1):
            a = i * 2.0 * math.pi / n
            vertex.addData3(0.0, math.cos(a), math.sin(a))
            vertex.addData3(1.0, math.cos(a), math.sin(a))
            normal.addData3(normalized(0.0, math.cos(a), math.sin(a)))
            normal.addData3(normalized(0.0, math.cos(a), math.sin(a)))
            texcoord.addData2f(float(i)/n, 0.0)
            texcoord.addData2f(float(i)/n, 1.0)
            if i > 0:
                tris.addVertices(2*i, 2*i-1, 2*i-2)
                tris.addVertices(2*i, 2*i+1, 2*i-1)

        cyl = Geom(vdata)
        cyl.addPrimitive(tris)
        geom = GeomNode('cylinder')
        geom.addGeom(cyl)
        node = NodePath(geom)
        return node

    @staticmethod
    def Plane(n):
        """ make a square of n x n vertices on the plane XY. The plane is between 0 and 1 in X and in Y """
        format = GeomVertexFormat.getV3n3cpt2()
        vdata = GeomVertexData('square', format, Geom.UHDynamic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        texcoord = GeomVertexWriter(vdata, 'texcoord')
        tris = GeomTriangles(Geom.UHDynamic)

        for i in range(n):
            for j in range(n):
                vertex.addData3(float(i)/(n-1), float(j)/(n-1), 0.0)
                normal.addData3(0.0, 0.0, 1.0)
                texcoord.addData2f(float(i)/(n-1), float(j)/(n-1))
                if i > 0 and j > 0:
                    p0 = i*n + j
                    p1 = (i-1)*n + j
                    p2 = (i-1)*n + j-1
                    p3 = i*n + j-1
                    tris.addVertices(p0, p1, p2)
                    tris.addVertices(p0, p2, p3)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        geom_node = GeomNode('square')
        geom_node.addGeom(geom)
        node = NodePath(geom_node)
        return node

    @staticmethod
    def Sphere(n):
        format = GeomVertexFormat.getV3n3cpt2()
        vdata = GeomVertexData('sphere', format, Geom.UHDynamic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        texcoord = GeomVertexWriter(vdata, 'texcoord')
        tris = GeomTriangles(Geom.UHDynamic)

        idp = 0
        for j in range(n+1):
            a = -0.5 * math.pi + float(j) * math.pi / n
            a_next = -0.5 * math.pi + float(j+1) * math.pi / n
            for i in range(2 * n+1):
                b = float(i) * 2.0 * math.pi / (2.0 * n)
                vertex.addData3(
                    math.cos(a) * math.cos(b),
                    math.cos(a) * math.sin(b),
                    math.sin(a))
                vertex.addData3(
                    math.cos(a_next) * math.cos(b),
                    math.cos(a_next) * math.sin(b),
                    math.sin(a_next))
                idp += 2
                normal.addData3(
                    math.cos(a) * math.cos(b),
                    math.cos(a) * math.sin(b),
                    math.sin(a))
                normal.addData3(
                    math.cos(a_next) * math.cos(b),
                    math.cos(a_next) * math.sin(b),
                    math.sin(a_next))
                texcoord.addData2f(float(i)/(2*n), float(j)/n)
                texcoord.addData2f(float(i)/(2*n), float(j+1)/n)
                if i > 0:
                    tris.addVertices(idp-1, idp-3, idp-2)
                    tris.addVertices(idp-2, idp-3, idp-4)

        sph = Geom(vdata)
        sph.addPrimitive(tris)
        geom = GeomNode('cylinder')
        geom.addGeom(sph)
        return NodePath(geom)

    @staticmethod
    def Grid(xmax, ymax, xstep, ystep, thickness, color=(1, 1, 1, 0), height=0.0):

        lines = LineSegs()
        lines.set_color(color)
        lines.set_thickness(thickness)

        for x in range(0, xmax+1, xstep):
            lines.move_to(x, 0, height)
            lines.draw_to(x, ymax, height)

        for y in range(0, ymax+1, ystep):
            lines.move_to(0, y, height)
            lines.draw_to(xmax, y, height)

        node = lines.create()
        return NodePath(node)

    @staticmethod
    def Cube():
        helper = VertexDataHelper()

        # This was adapated from a snippet found here
        # https://discourse.panda3d.org/t/panda3d-cube-geometry/2579
        # TODO: replace w/ a loop

        helper.vertex.addData3f(-0.50000, -0.50000, 0.50000)
        helper.texcoord.addData2f(0.26500, 0.67500)
        helper.normal.addData3f(0.00000, -0.00000, 1.00000)

        helper.vertex.addData3f(0.50000, 0.50000, 0.50000)
        helper.texcoord.addData2f(0.49000, 0.97500)
        helper.normal.addData3f(0.00000, -0.00000, 1.00000)

        helper.vertex.addData3f(-0.50000, 0.50000, 0.50000)
        helper.texcoord.addData2f(0.26500, 0.97500)
        helper.normal.addData3f(0.00000, -0.00000, 1.00000)

        helper.vertex.addData3f(-0.50000, -0.50000, 0.50000)
        helper.texcoord.addData2f(0.26500, 0.67500)
        helper.normal.addData3f(0.00000, -0.00000, 1.00000)

        helper.vertex.addData3f(0.50000, -0.50000, 0.50000)
        helper.texcoord.addData2f(0.49000, 0.67500)
        helper.normal.addData3f(0.00000, -0.00000, 1.00000)

        helper.vertex.addData3f(0.50000, 0.50000, 0.50000)
        helper.texcoord.addData2f(0.49000, 0.97500)
        helper.normal.addData3f(0.00000, -0.00000, 1.00000)

        helper.vertex.addData3f(-0.50000, -0.50000, -0.50000)
        helper.texcoord.addData2f(0.26500, 0.35000)
        helper.normal.addData3f(0.00000, -1.00000, -0.00000)

        helper.vertex.addData3f(0.50000, -0.50000, 0.50000)
        helper.texcoord.addData2f(0.49000, 0.65000)
        helper.normal.addData3f(0.00000, -1.00000, -0.00000)

        helper.vertex.addData3f(-0.50000, -0.50000, 0.50000)
        helper.texcoord.addData2f(0.26500, 0.65000)
        helper.normal.addData3f(0.00000, -1.00000, -0.00000)

        helper.vertex.addData3f(-0.50000, -0.50000, -0.50000)
        helper.texcoord.addData2f(0.26500, 0.35000)
        helper.normal.addData3f(0.00000, -1.00000, -0.00000)

        helper.vertex.addData3f(0.50000, -0.50000, -0.50000)
        helper.texcoord.addData2f(0.49000, 0.35000)
        helper.normal.addData3f(0.00000, -1.00000, -0.00000)

        helper.vertex.addData3f(0.50000, -0.50000, 0.50000)
        helper.texcoord.addData2f(0.49000, 0.65000)
        helper.normal.addData3f(0.00000, -1.00000, -0.00000)

        helper.vertex.addData3f(0.50000, 0.50000, -0.50000)
        helper.texcoord.addData2f(0.73500, 0.35000)
        helper.normal.addData3f(1.00000, 0.00000, 0.00000)

        helper.vertex.addData3f(0.50000, -0.50000, 0.50000)
        helper.texcoord.addData2f(0.51000, 0.65000)
        helper.normal.addData3f(1.00000, 0.00000, 0.00000)

        helper.vertex.addData3f(0.50000, -0.50000, -0.50000)
        helper.texcoord.addData2f(0.51000, 0.35000)
        helper.normal.addData3f(1.00000, 0.00000, 0.00000)

        helper.vertex.addData3f(0.50000, 0.50000, -0.50000)
        helper.texcoord.addData2f(0.73500, 0.35000)
        helper.normal.addData3f(1.00000, 0.00000, 0.00000)

        helper.vertex.addData3f(0.50000, 0.50000, 0.50000)
        helper.texcoord.addData2f(0.73500, 0.65000)
        helper.normal.addData3f(1.00000, 0.00000, 0.00000)

        helper.vertex.addData3f(0.50000, -0.50000, 0.50000)
        helper.texcoord.addData2f(0.51000, 0.65000)
        helper.normal.addData3f(1.00000, 0.00000, 0.00000)

        helper.vertex.addData3f(0.50000, 0.50000, -0.50000)
        helper.texcoord.addData2f(0.75500, 0.35000)
        helper.normal.addData3f(0.00000, 1.00000, 0.00000)

        helper.vertex.addData3f(-0.50000, 0.50000, 0.50000)
        helper.texcoord.addData2f(0.98000, 0.65000)
        helper.normal.addData3f(0.00000, 1.00000, 0.00000)

        helper.vertex.addData3f(0.50000, 0.50000, 0.50000)
        helper.texcoord.addData2f(0.75500, 0.65000)
        helper.normal.addData3f(0.00000, 0.00000, -1.00000)

        helper.vertex.addData3f(0.50000, 0.50000, -0.50000)
        helper.texcoord.addData2f(0.75500, 0.35000)
        helper.normal.addData3f(0.00000, 1.00000, 0.00000)

        helper.vertex.addData3f(-0.50000, 0.50000, -0.50000)
        helper.texcoord.addData2f(0.98000, 0.35000)
        helper.normal.addData3f(0.00000, 1.00000, 0.00000)

        helper.vertex.addData3f(-0.50000, 0.50000, 0.50000)
        helper.texcoord.addData2f(0.98000, 0.65000)
        helper.normal.addData3f(0.00000, 1.00000, 0.00000)

        helper.vertex.addData3f(-0.50000, 0.50000, 0.50000)
        helper.texcoord.addData2f(0.02000, 0.65000)
        helper.normal.addData3f(-1.00000, 0.00000, 0.00000)

        helper.vertex.addData3f(-0.50000, -0.50000, -0.50000)
        helper.texcoord.addData2f(0.24500, 0.35000)
        helper.normal.addData3f(-1.00000, 0.00000, 0.00000)

        helper.vertex.addData3f(-0.50000, -0.50000, 0.50000)
        helper.texcoord.addData2f(0.24500, 0.65000)
        helper.normal.addData3f(-1.00000, 0.00000, 0.00000)

        helper.vertex.addData3f(-0.50000, 0.50000, 0.50000)
        helper.texcoord.addData2f(0.02000, 0.65000)
        helper.normal.addData3f(-1.00000, 0.00000, 0.00000)

        helper.vertex.addData3f(-0.50000, 0.50000, -0.50000)
        helper.texcoord.addData2f(0.02000, 0.35000)
        helper.normal.addData3f(-1.00000, 0.00000, 0.00000)

        helper.vertex.addData3f(-0.50000, -0.50000, -0.50000)
        helper.texcoord.addData2f(0.24500, 0.35000)
        helper.normal.addData3f(-1.00000, 0.00000, 0.00000)

        helper.vertex.addData3f(0.50000, -0.50000, -0.50000)
        helper.texcoord.addData2f(0.49000, 0.32500)
        helper.normal.addData3f(0.00000, -1.00000, 0.00000)

        helper.vertex.addData3f(-0.50000, 0.50000, -0.50000)
        helper.texcoord.addData2f(0.26500, 0.02500)
        helper.normal.addData3f(0.00000, -1.00000, 0.00000)

        helper.vertex.addData3f(0.50000, 0.50000, -0.50000)
        helper.texcoord.addData2f(0.49000, 0.02500)
        helper.normal.addData3f(0.00000, -1.00000, 0.00000)

        helper.vertex.addData3f(0.50000, -0.50000, -0.50000)
        helper.texcoord.addData2f(0.49000, 0.32500)
        helper.normal.addData3f(0.00000, -1.00000, 0.00000)

        helper.vertex.addData3f(-0.50000, -0.50000, -0.50000)
        helper.texcoord.addData2f(0.26500, 0.32500)
        helper.normal.addData3f(0.00000, -1.00000, 0.00000)

        helper.vertex.addData3f(-0.50000, 0.50000, -0.50000)
        helper.texcoord.addData2f(0.26500, 0.02500)
        helper.normal.addData3f(0.00000, -1.00000, 0.00000)

        geom = Geom(helper.vdata)

        for i in range(0, 36, 3):
            for v in range(3):
                helper.tris.addVertex(i+v)
            helper.tris.closePrimitive()
            geom.addPrimitive(helper.tris)

        node = GeomNode('Cube')
        node.addGeom(geom)
        return NodePath(node)
