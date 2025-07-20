#!/usr/bin/env python3
"""Genesis-world 0.2.1 Examples: Creating 3D Robots Without URDF Issues

This file demonstrates various methods to create 3D robots in Genesis without
relying on URDF files, which can sometimes cause compatibility issues.
"""

import numpy as np
import genesis as gs


def example_1_simple_box_robot():
    """Create a simple robot using box primitives."""
    print("\n=== Example 1: Simple Box Robot ===")
    
    # Initialize Genesis
    gs.init(backend=gs.gpu)
    
    # Create scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, -9.81)
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -2.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=60,
        ),
        show_viewer=True,
    )
    
    # Add ground plane
    plane = scene.add_entity(gs.morphs.Plane())
    
    # Create robot body parts using boxes
    # Main body
    body = scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0, 0.5),
            size=(0.6, 0.4, 0.2),
        ),
    )
    
    # Add legs as separate boxes
    leg_positions = [
        (0.2, 0.15, 0.25),   # Front left
        (0.2, -0.15, 0.25),  # Front right
        (-0.2, 0.15, 0.25),  # Rear left
        (-0.2, -0.15, 0.25), # Rear right
    ]
    
    legs = []
    for x, y, z in leg_positions:
        leg = scene.add_entity(
            gs.morphs.Box(
                pos=(x, y, z),
                size=(0.05, 0.05, 0.4),
            ),
        )
        legs.append(leg)
    
    # Build and run
    scene.build()
    
    print("Box robot created! Running simulation...")
    for i in range(500):
        scene.step()
        
    return scene, body, legs


def example_2_sphere_and_cylinder_robot():
    """Create a robot using spheres and cylinders."""
    print("\n=== Example 2: Sphere and Cylinder Robot ===")
    
    gs.init(backend=gs.gpu)
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -2.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
    )
    
    # Ground
    plane = scene.add_entity(gs.morphs.Plane())
    
    # Robot body as sphere
    body = scene.add_entity(
        gs.morphs.Sphere(
            pos=(0, 0, 0.5),
            radius=0.3,
        ),
    )
    
    # Add smaller spheres as joints
    joints = []
    joint_positions = [
        (0.3, 0, 0.5),    # Front
        (-0.3, 0, 0.5),   # Back
        (0, 0.3, 0.5),    # Left
        (0, -0.3, 0.5),   # Right
    ]
    
    for x, y, z in joint_positions:
        joint = scene.add_entity(
            gs.morphs.Sphere(
                pos=(x, y, z),
                radius=0.1,
            ),
        )
        joints.append(joint)
    
    scene.build()
    
    print("Sphere robot created! Running simulation...")
    for i in range(500):
        scene.step()
        
    return scene, body, joints


def example_3_animated_box_robot():
    """Create an animated robot that bounces."""
    print("\n=== Example 3: Animated Bouncing Robot ===")
    
    gs.init(backend=gs.gpu)
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -2.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
    )
    
    # Ground
    plane = scene.add_entity(gs.morphs.Plane())
    
    # Robot parts
    body = scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0, 0.5),
            size=(0.4, 0.3, 0.15),
        ),
    )
    
    # Four legs
    legs = []
    leg_base_positions = [
        (0.15, 0.1, 0.4),
        (0.15, -0.1, 0.4),
        (-0.15, 0.1, 0.4),
        (-0.15, -0.1, 0.4),
    ]
    
    for x, y, z in leg_base_positions:
        leg = scene.add_entity(
            gs.morphs.Box(
                pos=(x, y, z),
                size=(0.04, 0.04, 0.3),
            ),
        )
        legs.append(leg)
    
    scene.build()
    
    print("Animated robot created! Running bouncing animation...")
    
    # Animation loop
    for i in range(1000):
        # Calculate bounce
        time = i * 0.01
        bounce_height = 0.1 * abs(np.sin(time * 3))
        
        # Update body position
        body.set_pos(np.array([0, 0, 0.5 + bounce_height]))
        
        # Update leg positions with phase offset
        for j, (leg, (x, y, z)) in enumerate(zip(legs, leg_base_positions)):
            phase = j * np.pi / 2
            leg_bounce = bounce_height * 0.7
            leg_sway = 0.02 * np.sin(time * 5 + phase)
            
            leg.set_pos(np.array([x + leg_sway, y, z + leg_bounce]))
        
        scene.step()
        
    return scene


def example_4_mesh_based_robot():
    """Example of loading a robot from mesh files (if available)."""
    print("\n=== Example 4: Mesh-Based Robot (Template) ===")
    
    # This is a template showing how to load from mesh files
    print("To load from mesh files, use:")
    print("""
    robot = scene.add_entity(
        gs.morphs.Mesh(
            file='path/to/robot.obj',  # or .ply, .stl, .glb, .gltf
            pos=(0, 0, 0),
            scale=1.0,
            fixed=True,  # Important: fix base to world
        ),
    )
    """)
    
    print("\nKey points for mesh-based robots:")
    print("- Mesh files should be in .obj, .ply, .stl, .glb, or .gltf format")
    print("- Use fixed=True to fix the base link to world")
    print("- Scale parameter can adjust the size")
    print("- Position and orientation can be set with pos and euler/quat")


def example_5_soft_robot():
    """Create a soft robot using MPM material."""
    print("\n=== Example 5: Soft Robot ===")
    
    gs.init(backend=gs.gpu)
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -2.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=40,
        ),
    )
    
    # Ground
    plane = scene.add_entity(gs.morphs.Plane())
    
    # Soft robot sphere
    soft_robot = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0, 0, 0.3),
            radius=0.2,
        ),
        material=gs.materials.MPM.Muscle(
            E=1e5,      # Young's modulus
            nu=0.3,     # Poisson's ratio
            rho=1000,   # Density
            model='neohooken',
        ),
    )
    
    scene.build()
    
    print("Soft robot created! Running simulation...")
    
    # Apply control signal
    for i in range(500):
        # Sine wave actuation
        actuation = 0.5 * (1 + np.sin(i * 0.05))
        soft_robot.set_actuation(actuation)
        scene.step()
        
    return scene, soft_robot


def example_6_mjcf_robot():
    """Example of loading robot from MJCF file (alternative to URDF)."""
    print("\n=== Example 6: MJCF Robot (Template) ===")
    
    print("To load from MJCF files, use:")
    print("""
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file='path/to/robot.xml',
            pos=(0, 0, 0),
            euler=(0, 0, 0),
        )
    )
    """)
    
    print("\nAdvantages of MJCF over URDF:")
    print("- MJCF files specify joint types including base-world connection")
    print("- Better compatibility with Genesis")
    print("- More features for physics simulation")


def main():
    """Run all examples."""
    print("Genesis-world 0.2.1: Creating 3D Robots Without URDF Issues")
    print("=" * 60)
    
    # Run examples
    try:
        # Example 1: Simple box robot
        scene1, body1, legs1 = example_1_simple_box_robot()
        
        # Example 2: Sphere robot
        scene2, body2, joints2 = example_2_sphere_and_cylinder_robot()
        
        # Example 3: Animated robot
        scene3 = example_3_animated_box_robot()
        
        # Example 4: Mesh template
        example_4_mesh_based_robot()
        
        # Example 5: Soft robot
        scene5, soft_robot = example_5_soft_robot()
        
        # Example 6: MJCF template
        example_6_mjcf_robot()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure genesis-world is installed: pip install genesis-world")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nKey takeaways:")
    print("1. Use primitive shapes (Box, Sphere) for simple robots")
    print("2. Use MJCF format instead of URDF for better compatibility")
    print("3. Use fixed=True for mesh-based robots to fix base")
    print("4. Soft robots can be created with MPM materials")
    print("5. Animation is done by updating positions with set_pos()")


if __name__ == "__main__":
    main()