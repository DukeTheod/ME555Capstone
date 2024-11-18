import trimesh
import pyvista as pv
from PIL import Image
import os

def render_stl_image(stl_file, output_image, resolution=(320, 240), angles=[0, 45, 90, 135, 180], color='green'):
    """
    Renders an STL model from different angles, applies color, and saves it as a 2D image.
    Arguments:
    - stl_file: path to the STL file.
    - output_image: path to save the rendered image.
    - resolution: resolution of the output image (width, height).
    - angles: List of angles (in degrees) to rotate the model and capture.
    - color: Color to apply to the model (e.g., 'red', 'green', 'blue').
    """
    
    # Load the STL file using trimesh
    mesh = trimesh.load_mesh(stl_file)

    # Convert the trimesh object to pyvista mesh
    mesh_pyvista = pv.wrap(mesh)

    # Create a PyVista plotter object
    plotter = pv.Plotter(off_screen=True)
    
    # Apply the color to the mesh
    plotter.add_mesh(mesh_pyvista, color=color)  # Apply color
    
    # Set the view size
    plotter.window_size = resolution  # Set resolution (width, height)
    plotter.set_background("white")  # Set background to white for better contrast
    
    # Iterate over angles to render the model from different perspectives
    for angle in angles:
        # Reset to an isometric view for consistency
        plotter.view_isometric()
        
        # Rotate the camera around the model by the specified angle
        plotter.camera.azimuth(angle)  # Rotate camera by azimuth (horizontal rotation)
        plotter.camera.elevation(30)  # Optionally, you can also adjust elevation for better view
        
        # Render the image
        output_image_with_angle = output_image.replace(".png", f"_{angle}.png")
        plotter.screenshot(output_image_with_angle)
        print(f"Rendered image saved to {output_image_with_angle}")

    # Optionally, open the last image
    img = Image.open(output_image_with_angle)
    img.show()

# Define the STL file path and output image path
stl_file = './dataset/part.stl'  # Path to your STL file in the dataset folder
output_image = './rendered_image.png'  # Image file to save the output in the current directory

# Render and save the image from multiple angles with color
render_stl_image(stl_file, output_image, color='green')  # You can change 'green' to any color you'd like