import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

inp = self.GetInput()
out = self.GetOutput()
out.ShallowCopy(inp)

# ✅ Correct time
info = inp.GetInformation()
t = info.Get(vtk.vtkDataObject.DATA_TIME_STEP())
t = t / 320

# ImageData geometry
origin = inp.GetOrigin()
spacing = inp.GetSpacing()
dims = inp.GetDimensions()

x = origin[0] + spacing[0] * np.arange(dims[0])
y = origin[1] + spacing[1] * np.arange(dims[1])
z = origin[2] + spacing[2] * np.arange(dims[2])

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Compute vector components
u_x_ex = np.sin(t) * np.sin(X + spacing[0]) * np.sin(Y) * np.sin(Z)
u_y_ex = np.sin(t) * np.cos(X) * np.cos(Y + spacing[1]) * np.cos(Z)
u_z_ex = np.sin(t) * np.cos(X) * np.sin(Y) * (np.cos(Z + spacing[2]) + np.sin(Z + spacing[2]))

# Magnitude
u_mag_exact = u_x_ex**2 + u_y_ex**2 + u_z_ex**2

# Convert to VTK arrays and add to output
vtk_x = numpy_to_vtk(u_x_ex.ravel(order='F'), deep=True)
vtk_x.SetName("u_x_ex")
out.GetPointData().AddArray(vtk_x)

vtk_y = numpy_to_vtk(u_y_ex.ravel(order='F'), deep=True)
vtk_y.SetName("u_y_ex")
out.GetPointData().AddArray(vtk_y)

vtk_z = numpy_to_vtk(u_z_ex.ravel(order='F'), deep=True)
vtk_z.SetName("u_z_ex")
out.GetPointData().AddArray(vtk_z)

vtk_mag = numpy_to_vtk(u_mag_exact.ravel(order='F'), deep=True)
vtk_mag.SetName("u_mag_exact")
out.GetPointData().AddArray(vtk_mag)

# Optionally, set active scalars
out.GetPointData().SetActiveScalars("u_mag_exact")

