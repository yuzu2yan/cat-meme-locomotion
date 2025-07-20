# Genesis-world 0.2.1: Best Practices for Creating 3D Robots Without URDF Issues

## Overview

Genesis-world is a powerful physics simulation platform for robotics. While it supports URDF files, there are several alternative approaches that can avoid common URDF compatibility issues.

## Common URDF Issues in Genesis

1. **Base Link Connection**: URDF files don't specify how the robot's base connects to the world, leading to floating robots
2. **Material Properties**: Genesis material specifications may differ from URDF standards
3. **Joint Limits**: Some URDF joint definitions may not translate perfectly
4. **File Path Issues**: Relative paths in URDF files can cause loading problems

## Alternative Approaches

### 1. Use Primitive Shapes

The simplest and most reliable approach is to build robots from primitive shapes:

```python
import genesis as gs

# Initialize
gs.init(backend=gs.gpu)

# Create scene
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.0, -2.0, 2.0),
        camera_lookat=(0.0, 0.0, 0.5),
    ),
)

# Add ground
plane = scene.add_entity(gs.morphs.Plane())

# Create robot from boxes
body = scene.add_entity(
    gs.morphs.Box(
        pos=(0, 0, 0.5),
        size=(0.6, 0.4, 0.2),
    ),
)

# Build scene
scene.build()
```

### 2. Use MJCF Format Instead

MJCF (MuJoCo XML) files have better Genesis compatibility:

```python
robot = scene.add_entity(
    gs.morphs.MJCF(
        file='robot.xml',
        pos=(0, 0, 0),
        euler=(0, 0, 0),
    )
)
```

### 3. Load Mesh Files Directly

For non-articulated robots, load mesh files directly:

```python
robot = scene.add_entity(
    gs.morphs.Mesh(
        file='robot.obj',  # Supports .obj, .ply, .stl, .glb, .gltf
        pos=(0, 0, 0),
        scale=1.0,
        fixed=True,  # Important: fixes base to world
    ),
)
```

### 4. Create Soft Robots

Genesis excels at soft body simulation:

```python
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
```

## Best Practices

### 1. Scene Setup

Always initialize Genesis properly:

```python
gs.init(backend=gs.gpu)  # or gs.cpu for CPU-only systems
```

### 2. Viewer Options

Configure viewer for better visualization:

```python
viewer_options=gs.options.ViewerOptions(
    camera_pos=(3.0, -2.0, 2.0),
    camera_lookat=(0.0, 0.0, 0.5),
    camera_fov=40,
    max_FPS=60,
)
```

### 3. Simulation Options

Set appropriate time step:

```python
sim_options=gs.options.SimOptions(
    dt=0.01,  # 10ms timestep
    gravity=(0, 0, -9.81),
)
```

### 4. Animation Techniques

For dynamic robots, update positions in the simulation loop:

```python
for i in range(1000):
    # Calculate new position
    new_pos = calculate_position(i)
    
    # Update entity position
    robot.set_pos(new_pos)
    
    # Step simulation
    scene.step()
```

### 5. Joint Control (Without URDF)

Create articulated robots by connecting primitives:

```python
# Create connected boxes for articulated motion
upper_arm = scene.add_entity(gs.morphs.Box(...))
lower_arm = scene.add_entity(gs.morphs.Box(...))

# Update positions to simulate joint motion
angle = np.sin(time)
lower_arm.set_pos(calculate_joint_position(angle))
```

## Working Example: Quadruped Robot

Here's a complete example of a simple quadruped without URDF:

```python
import numpy as np
import genesis as gs

def create_quadruped():
    gs.init(backend=gs.gpu)
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -2.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
    )
    
    # Ground
    plane = scene.add_entity(gs.morphs.Plane())
    
    # Body
    body = scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0, 0.5),
            size=(0.6, 0.4, 0.2),
        ),
    )
    
    # Legs
    leg_positions = [
        (0.2, 0.15, 0.3),   # Front left
        (0.2, -0.15, 0.3),  # Front right
        (-0.2, 0.15, 0.3),  # Rear left
        (-0.2, -0.15, 0.3), # Rear right
    ]
    
    legs = []
    for pos in leg_positions:
        leg = scene.add_entity(
            gs.morphs.Box(
                pos=pos,
                size=(0.05, 0.05, 0.4),
            ),
        )
        legs.append(leg)
    
    scene.build()
    
    # Animate
    for i in range(1000):
        time = i * 0.01
        
        # Body bounce
        bounce = 0.05 * np.sin(time * 3)
        body.set_pos(np.array([0, 0, 0.5 + bounce]))
        
        # Leg movement
        for j, (leg, base_pos) in enumerate(zip(legs, leg_positions)):
            phase = j * np.pi / 2
            x, y, z = base_pos
            
            # Add gait pattern
            leg_lift = 0.05 * max(0, np.sin(time * 5 + phase))
            leg.set_pos(np.array([x, y, z + leg_lift]))
        
        scene.step()

if __name__ == "__main__":
    create_quadruped()
```

## Troubleshooting

### Issue: Robot Falls Through Ground
- Ensure collision detection is enabled
- Check material properties
- Verify entity positions don't overlap

### Issue: Simulation Too Slow
- Reduce polygon count in meshes
- Use primitive shapes instead of complex meshes
- Adjust simulation timestep

### Issue: Jerky Motion
- Decrease timestep (dt)
- Smooth position updates
- Check for discontinuities in motion data

## Resources

- Genesis Documentation: https://genesis-world.readthedocs.io/
- Genesis GitHub: https://github.com/Genesis-Embodied-AI/Genesis
- Example Scripts: See `genesis_robot_examples.py` in this repository

## Summary

While Genesis supports URDF files, using alternative approaches like primitive shapes, MJCF files, or direct mesh loading can avoid compatibility issues and provide more control over robot creation and simulation.