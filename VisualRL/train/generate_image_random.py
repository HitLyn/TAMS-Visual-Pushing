from robogym.envs.push.push_env import make_env
PATH = '/homeL/cong/HitLyn/Visual-Pushing/images/all_objects_random/'
NUM_IMAGES = 40000
def main():
    env = make_env()
    for i in range(NUM_IMAGES):
        env.reset()
        with env.mujoco_simulation.hide_target():
            array = env.render(mode = "rgb_array")
        name = PATH + "{:0>5d}.png".format(i)
        plt.imsave(name,array,format = 'png')


if __name__ == '__main__':
    from mujoco_py import GlfwContext
    import matplotlib.pyplot as plt
    GlfwContext(offscreen=True)
    main()
