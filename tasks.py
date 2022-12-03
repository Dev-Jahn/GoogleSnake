from invoke import task


@task
def train(c, method, nenv=10, device='cuda', postfix='base'):
    if method == 'ppo':
        from test.PPO_train import main
    elif method == 'bmg':
        from test.BMG_train import main
    # elif method == 'maml':
    #     from maml_rl import main
    # elif method == 'reptile':
    #     from reptile import main
    else:
        raise ValueError('Unknown method: {}'.format(method))
    main(n_env=int(nenv), device=device, postfix=postfix)


@task
def eval(c, method):
    if method == 'ppo':
        from test.PPO_eval import main
    # elif method == 'maml':
    #     from maml_rl import main
    # elif method == 'reptile':
    #     from reptile import main
    else:
        raise ValueError('Unknown method: {}'.format(method))
    main()


@task
def test(c, **kwargs):
    from test import test_env
    test_env.main()
