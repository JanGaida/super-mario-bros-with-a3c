# Generel
import timeit

# Torch
import torch as T
import torch.nn.functional as F

from torch.distributions import Categorical
from collections import deque

# TensorboardX
from tensorboardX import SummaryWriter

# Hilfsklassen & -Funktionen
from bin.enviorment import make_training_enviorment
from bin.model import ActorCriticModel

def dispatch_training(idx, args, global_model, optimizer, should_save = False, trained_episodes = 0):
    """Die Worker Aufgabe für ein Training"""
    try:

        # Bereite Torch-Multiprocessing vor
        T.manual_seed(args.torch_seed + idx)
        cuda = args.cuda

        # Falls Verbose
        verbose = args.verbose
        if verbose: start_time = timeit.default_timer()

        # Tensorboard
        writer = SummaryWriter(args.logdir)

        # Lokales Enviorment
        env, num_states, num_actions = make_training_enviorment(args)

        # Lokales Model
        local_model = ActorCriticModel(num_states, num_actions)
        if cuda: local_model.cuda() # GPU-Support
        local_model.train() # Trainings-Flag

        # Loop-Var
        local_state = T.from_numpy( env.reset() )
        if cuda: local_state = local_state.cuda() # GPU-Support
        local_done = True # ob das gym-level abgeschlossen ist
        local_episode = trained_episodes # aktuelle worker-episode
        local_step = 0 # aktueller worker-step
        local_reward = 0 # aktueller worker-reward
        total_loss = 0

        # Loop-Const
        episode_save_interval = args.episode_save_interval # der Speicherinterval
        model_save_name = args.model_save_name # der Name des gespeichert Model
        modeldir = args.modeldir # der Pfad in dem das Model gespeichert wird
        world = args.world
        stage = args.stage
        rversion = args.rversion
        max_local_steps = args.max_local_steps
        max_global_steps = args.max_global_steps
        discount_gamma = args.discount_gamma
        tau = args.tau
        beta = args.beta
        verbose_every_episode = args.verbose_every_episode
        num_parallel_trainings_threads = args.num_parallel_trainings_threads

        # Für Verbose vorzeitige init
        if verbose:
            ep_rewards = [0]

        # unendliche Trainings-Loop
        while True:

            # Überprüfe ob gespeichert werden soll
            if should_save and local_episode % episode_save_interval == 0 and not local_episode == trained_episodes:
                T.save(global_model.state_dict(), "{}/{}_world{}_stage{}_ver{}__ep{}_x_{}.pt".format(modeldir, model_save_name, world, stage, rversion, local_episode, num_parallel_trainings_threads))
                if verbose: print("\nWorker {: 2d} :: Training    ---    globales Model gespeichert\n".format(idx))

            # Nächste Episode
            local_episode += 1
            if verbose and local_episode % verbose_every_episode == 0 and not local_episode == 0: 
                latest_sum_reward = sum(ep_rewards)
                latest_avg_reward = latest_sum_reward / len(ep_rewards)
                print("Worker {: 2d} :: Training    ---    lokale Episode {:>7}    ---    lokale Avg.Reward {:>10.3f}    ---    lokale Sum.Reward {:>10.3f}    ---    Loss {:>12.2f}"\
                    .format(idx, local_episode, latest_avg_reward, latest_sum_reward, total_loss.item()))

            # Gewichte aus dem globalen Model laden
            local_model.load_state_dict(global_model.state_dict())

            # Episoden Tensor
            if local_done: # Neue Tensor erzeugen falls benötigt
                h_0 = T.zeros( (1, 512), dtype = T.float)
                c_0 = T.zeros( (1, 512), dtype = T.float)
            else: # Wiederverwenden
                h_0 = h_0.detach()
                c_0 = c_0.detach()
            if cuda: # CUDA-Support
                h_0 = h_0.cuda()
                c_0 = c_0.cuda()

            # Episoden-Var
            ep_policies = []
            ep_judgment = []
            ep_rewards = []
            ep_entropies = []

            # Episoden-Loop
            for _ in range(max_local_steps):
                local_step += 1

                # Model
                action_logit_probability, action_judgement, h_0, c_0 = local_model(local_state, h_0, c_0)

                # Policies
                policy = F.softmax(action_logit_probability, dim = 1)
                log_policy = F.log_softmax(action_logit_probability, dim = 1)

                # Entropie
                entropy = (policy * log_policy).sum(1, keepdim = True)

                # Entscheidung für eine Aktion mit logits-Algorithmus
                action = Categorical(policy).sample().item()
                
                # Führe Aktion aus
                local_state, local_reward, local_done, _ = env.step(action)
                local_state = T.from_numpy(local_state)
                if cuda: local_state = local_state.cuda() # GPU-Support

                # Erfahrungen aufbauen
                ep_policies.append(log_policy[0, action])
                ep_judgment.append(action_judgement)
                ep_rewards.append(local_reward)
                ep_entropies.append(entropy)

                # Überprüft ob noch Schritte getan werden können
                if local_step > max_global_steps:
                    # Beginnt eine neue Episode
                    local_done = True

                # Wenn die Episode abgeschlossen ist...
                if local_done:
                    # Zurücksetzten der Steps & des Enviorments
                    local_step = 0
                    local_state = T.from_numpy( env.reset() )
                    if cuda: local_state = local_state.cuda() # GPU-Support

                # Überprüft ob die nächste Episode gestartet werden soll
                if local_done:
                    break

            # Bewertung
            R = T.zeros((1, 1), dtype=T.float)
            if cuda: R = R.cuda() # GPU-Support

            if not local_done: 
                # Bewertung einholen für Runs die nicht abgeschlossen wurden
                _, R, _, _ = local_model(local_state, h_0, c_0)

            gae = T.zeros((1,1), dtype=T.float)
            if cuda: gae = gae.cuda()

            # Loss's
            actor_loss = 0
            critic_loss = 0
            entropy_loss = 0
            next_value = R

            # Loope alle Erfahrungen rückwärts (!)
            for judgment, log_policy, reward, entropy in list( zip(ep_judgment, ep_policies, ep_rewards, ep_entropies) )[::-1]:
                # Berechnung nach Boltzmann-Policy (?)
                gae = gae * discount_gamma * tau
                gae = gae + reward + discount_gamma * next_value.detach() - judgment.detach()
                next_value = judgment
                actor_loss = actor_loss + log_policy * gae
                R = R * discount_gamma + reward
                critic_loss = critic_loss + (R - judgment) ** 2 / 2
                entropy_loss = entropy_loss + entropy

            total_loss = -actor_loss + critic_loss - beta * entropy_loss

            # Tensorboard
            writer.add_scalar("Training_Worker_{}_Loss".format(idx), total_loss, local_episode)

            # Vor der Backpropagation alle Gradienten auf 0 setzen
            optimizer.zero_grad()

            # Backpropagation
            total_loss.backward()

            # Loope die Gradienten des globalen & lokalen Models
            for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                if global_param.grad is not None:
                    # Wenn im globalen ein Gradient ist
                    break
                # Wenn nicht übernehm das locale Gradient
                global_param._grad = local_param.grad

            # Übernehme die Gradienten wieder
            optimizer.step()

            # Finally, wenn 
            if local_episode == int(max_global_steps / max_local_steps):
                if verbose:
                    end_time = timeit.default_timer()
                    print("Worker {: 2d} :: Training    ---    nach {:.2f} s abgeschlossen".format(idx,(end_time - start_time)))
                else:
                    print("Worker {: 2d} :: Training    ---    abgeschlossen".format(idx))
                # Fertig
                return

    except KeyboardInterrupt:
        if verbose:
            end_time = timeit.default_timer()
            print("Worker {: 2d} :: Training    ---    Laufzeit {:.2f} s    ---    EXIT OK".format(idx,(end_time - start_time)))
        else:
            print("Worker {: 2d} :: Training    ---    EXIT OK".format(idx))
        # Fertig
        return


def dispatch_testing(idx, args, global_model):
    """Die Worker Aufgabe für ein Testing"""
    try:
        # Bereite Torch-Multiprocessing vor
        T.manual_seed(args.torch_seed + idx)

        # Enviorment initialisieren
        env, num_states, num_actions = make_training_enviorment(args)

        # Lokales Model
        local_model = ActorCriticModel(num_states, num_actions)
        local_model.eval() # Evaluation-Flag

        # Loop-Vars
        local_done = True
        local_step = 0
        local_state = T.from_numpy( env.reset() )
        actions = deque(maxlen = args.max_actions)
        max_global_steps = args.max_global_steps
        # Loop-Const

        # Testing-Loop
        while True:
            # Step hochzählen
            local_step += 1

            # Model wiederladen wenn Run abgeschlossen
            if local_done:
                #print("Runner {: 2d} :: Training    ---    Lade Globales Model nach".format(idx))
                local_model.load_state_dict(global_model.state_dict())

            # Ohne Gradienten-Berrechnung
            with T.no_grad():
                if local_done: # Neue Tensor erzeugen falls benötigt
                    h_0 = T.zeros((1, 512), dtype=T.float)
                    c_0 = T.zeros((1, 512), dtype=T.float)
                else: # Ansonsten wiederverwenden
                    h_0 = h_0.detach()
                    c_0 = c_0.detach()

            # Model
            action_logit_probability, action_judgement, h_0, c_0 = local_model(local_state, h_0, c_0)

            # Policy
            policy = F.softmax(action_logit_probability, dim=1)

            # Action
            action = T.argmax(policy).item()

            # Action durchführen
            local_state, local_reward, local_done, _ = env.step(action)
            env.render()

            # Aktion merken
            actions.append(action)

            # Wenn max_global_steps erreicht wurde oder wenn die max_actions erreich wurden ...
            if local_step > max_global_steps or actions.count(actions[0]) == actions.maxlen:
                #print("Runner {: 2d} :: Training    ---    Aktionslimit erreicht".format(idx))
                # .. neustarten
                local_done = True

            # Überprüft ob das Enviroment zrückgesetzt werden soll
            if local_done:
                # Variablen zurückstetzten
                local_step = 0
                actions.clear()
                # Env zurückstetzen
                local_state = env.reset()

            # zu numpy
            local_state = T.from_numpy(local_state)

    except KeyboardInterrupt:
        print("Runner {: 2d} :: Training    ---    EXIT OK".format(idx))
        return
