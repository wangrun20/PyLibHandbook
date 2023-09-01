import random
import math

from isaacgym import gymapi, gymtorch


def print_asset_info(asset, name):
    gym = gymapi.acquire_gym()
    print("======== Asset info %s: ========" % name)
    num_bodies = gym.get_asset_rigid_body_count(asset)
    num_joints = gym.get_asset_joint_count(asset)
    num_dofs = gym.get_asset_dof_count(asset)
    print('Got %d bodies, %d joints, and %d DOFs' %
          (num_bodies, num_joints, num_dofs))

    # Iterate through bodies
    print("Bodies:")
    for i in range(num_bodies):
        name = gym.get_asset_rigid_body_name(asset, i)
        print(" %2d: '%s'" % (i, name))

    # Iterate through joints
    print("Joints:")
    for i in range(num_joints):
        name = gym.get_asset_joint_name(asset, i)
        joint_type = gym.get_asset_joint_type(asset, i)
        type_name = gym.get_joint_type_string(joint_type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))

    # iterate through degrees of freedom (DOFs)
    print("DOFs:")
    for i in range(num_dofs):
        name = gym.get_asset_dof_name(asset, i)
        dof_type = gym.get_asset_dof_type(asset, i)
        type_name = gym.get_dof_type_string(dof_type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))


def main():
    # 这样获得的gym都是同一个实例, 它们的id都是相同的
    gym = gymapi.acquire_gym()

    # get default set of parameters
    sim_params = gymapi.SimParams()
    # set common parameters
    sim_params.dt = 1 / 60  # 与仿真环境交互的频率
    sim_params.enable_actor_creation_warning = True
    sim_params.substeps = 2  # 物理仿真时间步长为dt/substeps
    sim_params.up_axis = gymapi.UP_AXIS_Z  # 指定竖直向上的轴
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)  # 指定重力
    sim_params.num_client_threads = 0
    sim_params.use_gpu_pipeline = True  # if False, the tensors returned by Gym will reside on the CPU
    # set PhysX-specific parameters
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4  # PhysX solver position iterations count. default 4
    sim_params.physx.num_velocity_iterations = 1  # PhysX solver velocity iterations count. default 1
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.rest_offset = 0.01
    # create sim with these parameters
    # args: 计算GPU id, 渲染GPU id, 物理引擎, 仿真参数
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    # configure the ground plane (平地)
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)  # 平地的法向量. 对于UP_AXIS_Z, 平地的法向量为(0, 0, 1)
    plane_params.distance = 0  # 平面距离原点的距离
    plane_params.static_friction = 1  # 静态摩擦系数
    plane_params.dynamic_friction = 1  # 动态摩擦系数
    plane_params.restitution = 0  # 恢复系数, 控制与地面碰撞的弹性(反弹量)
    # create the ground plane
    gym.add_ground(sim, plane_params)

    asset_root = './assets/wheeled_dog'
    asset_file = 'urdf/robot.urdf'
    asset_options = gymapi.AssetOptions()
    asset_options.angular_damping = 0.5
    asset_options.armature = 0.0
    asset_options.collapse_fixed_joints = False
    asset_options.convex_decomposition_from_submeshes = False  # 允许加载子网格来表示凸分解
    asset_options.default_dof_drive_mode = 0
    asset_options.density = 1000.0
    asset_options.disable_gravity = False
    asset_options.enable_gyroscopic_forces = True
    asset_options.fix_base_link = False  # 把机器人基座固定在出生点
    asset_options.flip_visual_attachments = False  # visual材质可能需要翻转为y-up/z-up
    asset_options.linear_damping = 0.0
    asset_options.max_angular_velocity = 64.0
    asset_options.max_linear_velocity = 1000.0
    asset_options.mesh_normal_mode = gymapi.FROM_ASSET  # 法线修正方式
    asset_options.min_particle_mass = 1e-12
    asset_options.override_com = False  # isaac gym根据实体的碰撞箱来计算质心, 而不使用urdf提供的值
    asset_options.override_inertia = False  # isaac gym根据实体的碰撞箱来计算惯量, 而不使用urdf提供的值
    asset_options.replace_cylinder_with_capsule = False
    asset_options.slices_per_cylinder = 20
    asset_options.tendon_limit_stiffness = 1.0
    asset_options.thickness = 0.02
    asset_options.use_mesh_materials = False  # 使用网格作为材质
    asset_options.use_physx_armature = True
    asset_options.vhacd_enabled = False  # 使用V-HACD进行mesh凸分解
    asset_options.vhacd_params.resolution = 300000  # V-HACD参数
    asset_options.vhacd_params.max_convex_hulls = 10  # V-HACD参数
    asset_options.vhacd_params.max_num_vertices_per_ch = 64  # V-HACD参数
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    print_asset_info(asset, 'WheeledDog')

    # set up the env grid
    num_envs = 16
    envs_per_row = 8
    env_spacing = 2.0
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

    # cache some common handles for later use
    envs = []
    actor_handles = []

    # create and populate the environments
    for i in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
        envs.append(env)

        height = random.uniform(1.0, 2.5)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, height)
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.0 * math.pi)

        # actor是asset的一个实例, 必须从属于某一个env. 注意, 最好一次性添加完env中的所有actor, 再创建下一个环境
        # args: env, asset, actor在env坐标系中的出生点位姿, actor name, collision_group, collision_filter
        #   只有相同collision_group的物体才会发生碰撞, 特别地, collision_group为-1表示会与所有组碰撞
        #   一般情况下, 不同env中的actor的collision_group应该不同
        #   collision_filter是一个bit mask. 如果两实体的collision_filter have a common bit set, 则它俩不会碰撞
        actor_handle = gym.create_actor(env, asset, pose, 'WheeledDog', i, 1)
        actor_handles.append(actor_handle)

        # actor DoF属性. 一般在urdf中已经定义, 但也可以override
        props = gym.get_actor_dof_properties(env, actor_handle)
        # props['hasLimits'].fill(True)  # DOF pos是否有限制. 但属性似乎无法override
        props['lower'].fill(-1.0)  # limit下界
        props['upper'].fill(1.0)  # limit上界
        # 驱动模式, 有以下三种. 单位均为标准单位: m, rad, m/s, rad/s, N, Nm
        #   DOF_MODE_NONE. joints move freely within their range of motion
        #   DOF_MODE_EFFORT. we need to apply efforts to the DOF using apply_actor_dof_efforts every frame
        #       example:
        #           efforts = np.full(num_dofs, 100.0).astype(np.float32)
        #           gym.apply_actor_dof_efforts(env, actor_handle, efforts)
        #   DOF_MODE_POS. engage a PD controller that can be tuned using stiffness and damping parameters
        #       the controller will apply DOF forces that are proportional to posError * stiffness + velError * damping
        #       typically with both stiffness and damping set to non-zero values
        #       example:
        #           targets = np.zeros(num_dofs).astype('f')
        #           gym.set_actor_dof_position_targets(env, actor_handle, targets)
        #   DOF_MODE_VEL. torques applied by the PD controller will be proportional to the damping parameter
        #       the stiffness parameter should be set to zero
        #       example:
        #           vel_targets = np.random.uniform(-math.pi, math.pi, num_dofs).astype('f')
        #           gym.set_actor_dof_velocity_targets(env, actor_handle, vel_targets)
        # unlike efforts, position and velocity targets don’t need to be set every frame, only when changing targets
        # 注意, apply effort或set pos/vel都需要传入num_dofs长度的ndarray, 而只有对应drive mode的DOF才会接受更改, 其余DOF不变
        # 也可以单独控制某一个DOF, example:
        #   dof_handle = gym.find_actor_dof_handle(env, actor, 'dof_name')
        #   gym.set_dof_target_position(env, dof_handle, 11.16)
        props['driveMode'].fill(gymapi.DOF_MODE_NONE)
        props['stiffness'].fill(0.0)  # 刚度
        props['damping'].fill(0.0)  # 阻尼
        props['velocity'].fill(100.0)  # 最大速度
        props['effort'].fill(100.0)  # 最大力或力矩
        props['friction'].fill(0.0)  # 摩擦
        props['armature'].fill(0.0)  # ?
        gym.set_actor_dof_properties(env, actor_handle, props)

        # body的运动学状态. 可以提取all, 也可以只要位置或者速度
        # rigid body states include position (Vec3), orientation (Quat), linear velocity (Vec3), angular velocity (Vec3)
        # 可以获得actor, env, sim的states
        body_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_ALL)
        body_states = gym.get_env_rigid_body_states(env, gymapi.STATE_POS)
        body_states = gym.get_sim_rigid_body_states(sim, gymapi.STATE_VEL)
        # body_states['pose']               all poses (position and orientation)
        # body_states['pose']['p']          all positions (Vec3: x, y, z)
        # body_states['pose']['p']['x']     all x coordinates (ndarray)
        # body_states['pose']['r']          all orientations (Quat: x, y, z, w)
        # body_states['pose']['r']['w']     all w of Quat (ndarray)
        # body_states['vel']                all velocities (linear and angular)
        # body_states['vel']['linear']      all linear velocities (Vec3: x, y, z)
        # body_states['vel']['angular']     all angular velocities (Vec3: x, y, z)

        # 也可以设置运动状态
        # gym.set_actor_rigid_body_states(env, actor_handle, body_states, gymapi.STATE_ALL)
        # gym.set_env_rigid_body_states(env, body_states, gymapi.STATE_POS)
        # gym.set_sim_rigid_body_states(sim, body_states, gymapi.STATE_VEL)

        # 查询body states array中某一特定body的index
        i1 = gym.find_actor_rigid_body_index(env, actor_handle, 'body_name', gymapi.DOMAIN_ACTOR)
        i2 = gym.find_actor_rigid_body_index(env, actor_handle, 'body_name', gymapi.DOMAIN_ENV)
        i3 = gym.find_actor_rigid_body_index(env, actor_handle, 'body_name', gymapi.DOMAIN_SIM)

        # 类似的, 也可以对DOF的运动学状态操作
        dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)
        gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)
        i1 = gym.find_actor_dof_index(env, actor_handle, 'dof_name', gymapi.DOMAIN_ACTOR)
        i2 = gym.find_actor_dof_index(env, actor_handle, 'dof_name', gymapi.DOMAIN_ENV)
        i3 = gym.find_actor_dof_index(env, actor_handle, 'dof_name', gymapi.DOMAIN_SIM)

    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)

    # tensor API example
    gym.prepare_sim(sim)  # 使用tensor API的前置步骤
    _root_tensor = gym.acquire_actor_root_state_tensor(sim)
    # get所有actor的state of root body (每个actor只有一个root body). 只需acquire一次, 之后refresh即可
    # shape为(num_actors, 13), 3 for pos, 4 for Quat, 3 for linear vel, 3 for angular vel, 13 in total
    # in-place的更改root_tensor的值, 会使得actor瞬间改变运动状态
    root_tensor = gymtorch.wrap_tensor(_root_tensor)

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, 'exit')
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ENTER, 'track')

    while not gym.query_viewer_has_closed(viewer):
        # step the physics
        gym.simulate(sim)
        gym.refresh_actor_root_state_tensor(sim)  # update the contents of root_tensor with the latest state
        gym.fetch_results(sim, True)

        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == 'exit' and evt.value > 0:
                print('esc')
            elif evt.action == 'track' and evt.value > 0:
                print('enter')

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        # If the simulation is running slower than real time, this statement will have no effect.
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == '__main__':
    main()
