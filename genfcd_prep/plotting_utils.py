import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


def plot3Dcube_outer_layer(mat, Xvec, Yvec, Zvec, cmap='viridis',
                          title='', figsize=(12, 6),
                          vmin=None, vmax=None,
                          xlabel='x', ylabel='y', zlabel='z', vlabel='v'):
    ''' 
    Plot only the outer layer of a 3D cube
    
    Parameters:
    -----------
    mat : (Nx, Ny, Nz) 3D array of scalars indexed by x, y, z 
    Xvec : (Nx,) coordinates of the x axis
    Yvec : (Ny,) coordinates of the y axis
    Zvec : (Nz,) coordinates of the z axis
    '''
    # Input validation
    if Xvec[1]<=Xvec[0] or Yvec[1]<=Yvec[0] or Zvec[1]<=Zvec[0]:
        raise Exception('Coordinate vectors must be in ascending order')

    if (Xvec.shape[0]!=mat.shape[0] or Yvec.shape[0]!=mat.shape[1] or 
        Zvec.shape[0]!=mat.shape[2]):
        raise Exception(f'Coordinate vector shapes do not match data dimensions')

    # Create mask for outer layer only
    nx, ny, nz = mat.shape
    mask = np.zeros_like(mat, dtype=bool)
    
    # Set the outer faces to True
    mask[0, :, :] = mask[-1, :, :] = True  # x faces
    mask[:, 0, :] = mask[:, -1, :] = True  # y faces
    mask[:, :, 0] = mask[:, :, -1] = True  # z faces
    
    # Extend coordinates to include boundaries
    Xvec2 = np.concatenate([Xvec, [Xvec[-1]+Xvec[1]-Xvec[0]]])
    Yvec2 = np.concatenate([Yvec, [Yvec[-1]+Yvec[1]-Yvec[0]]])
    Zvec2 = np.concatenate([Zvec, [Zvec[-1]+Zvec[1]-Zvec[0]]])
    Xmat, Ymat, Zmat = np.meshgrid(Xvec2, Yvec2, Zvec2, indexing='ij')

    # Set color scale
    if vmin is None: vmin = mat.min()
    if vmax is None: vmax = mat.max()

    # Create color array
    cmap = plt.colormaps.get_cmap(cmap)
    cmap_array = cmap((mat-vmin)/(vmax-vmin))
    colors = np.zeros(mat.shape + (4,))
    colors[..., 0:3] = cmap_array[..., 0:3]
    colors[..., 3] = 1

    # Create plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([Xvec2[0], Xvec2[-1]])
    ax.set_ylim([Yvec2[0], Yvec2[-1]])
    ax.set_zlim([Zvec2[0], Zvec2[-1]])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    
    # Adjust the aspect ratio to make the cube taller
    ax.set_box_aspect([1, 1, 2])  # Make z-axis twice as tall
    
    ax.voxels(Xmat, Ymat, Zmat,
              mask,
              facecolors=colors,
              edgecolors=colors,
              linewidth=0.5)    
    
    # Add colorbar
    m = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    m.set_array([])
    if title: ax.set_title(title)
    plt.colorbar(m, ax=ax, shrink=0.4, pad=0.1, label=vlabel)
    plt.tight_layout()
    plt.show()
