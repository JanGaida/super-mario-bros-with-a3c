# Generel
import timeit, os
from datetime import datetime

# Torch
import torch as T
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

# Tensorboard
from tensorboardX import SummaryWriter

# Hilfsklassen & -Funktionen
from bin.enviorment import make_training_enviorment
from bin.model import ActorCriticModel


def dispatch_training(idx, args, global_model, optimizer, should_save, trained_episodes, summarywriter_path):
    """Die Worker Aufgabe für ein Training"""
    
    try:
        #summarywriter = SummaryWriter(summarywriter_path)

        # Bereite Torch-Multiprocessing vor
        T.manual_seed(args.torch_seed + idx)
        cuda = args.cuda

        # Falls Verbose
        verbose = args.verbose
        if verbose: start_time = timeit.default_timer()

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

        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
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
        absolute_max_training_steps = args.absolute_max_training_steps

        # Für Verbose vorzeitige init
        if verbose:
            ep_rewards = [0]
            loop_time_0 = timeit.default_timer()

        # Für das Speichern
        specific_modeldir = "{}/{}_world{}_stage{}_ver{}".format(modeldir, model_save_name, world, stage, rversion)
        if not os.path.isdir(specific_modeldir):
            os.mkdir(specific_modeldir)

        # unendliche Trainings-Loop
        while True:

            # Überprüfe ob gespeichert werden soll
            if should_save and local_episode % episode_save_interval == 0 and not local_episode == trained_episodes:
                T.save(global_model.state_dict(), "{}/ep{}_x_{}.pt".format(specific_modeldir, local_episode, num_parallel_trainings_threads))
                if verbose: print("\n{} :: Worker {: 2d}    ---    globales Model erfolgreich gespeichert\n".format(datetime.now().strftime("%H:%M:%S"), idx))

            # Nächste Episode
            local_episode += 1
            if verbose and local_episode % verbose_every_episode == 0 and not local_episode == 0:
                latest_sum_reward = sum(ep_rewards)
                latest_avg_reward = latest_sum_reward / len(ep_rewards)
                loop_time_1 = timeit.default_timer()
                print("{} :: Worker {: 2d}  |  EP {:>6} ({:>4.2f} ep/s)  |  Avg-RW {:>6.2f}  | Sum-RW {:>8.1f}  |  A-Loss {:>8.1f}  |  C-Loss {:>8.1f}  |  E-Loss {:>8.1f}  |  Loss {:>8.1f}".format(
                    datetime.now().strftime("%H:%M:%S"), idx, local_episode, ((loop_time_1 - loop_time_0)/verbose_every_episode), latest_avg_reward, latest_sum_reward, actor_loss.item(), critic_loss.item(), entropy_loss.item(), total_loss.item())
                )
                loop_time_0 = loop_time_1

            # Gewichte aus dem globalen Model laden
            local_model.load_state_dict(global_model.state_dict())

            # Episoden Tensor
            if local_done: # Neue Tensor erzeugen falls benötigt
                hx = T.zeros( (1, 512), dtype = T.float)
                cx = T.zeros( (1, 512), dtype = T.float)
            else: # Wiederverwenden
                hx = hx.detach()
                cx = cx.detach()
            if cuda: # CUDA-Support
                hx = hx.cuda()
                cx = cx.cuda()

            # Episoden-Var
            ep_policies = []
            ep_judgment = []
            ep_rewards = []
            ep_entropies = []

            # Episoden-Loop
            for _ in range(max_local_steps):
                local_step += 1

                # Model
                action_logit_probability, action_judgement, hx, cx = local_model(local_state, hx, cx)

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
                _, R, _, _ = local_model(local_state, hx, cx)

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
            #summarywriter.add_scalar("Worker-{}/actor_loss".format(idx), actor_loss.item(), local_episode)
            #summarywriter.add_scalar("Worker-{}/critic_loss".format(idx), critic_loss.item(), local_episode)
            #summarywriter.add_scalar("Worker-{}/total_loss".format(idx), total_loss.item(), local_episode)

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
            if local_episode == absolute_max_training_steps:
                if verbose:
                    end_time = timeit.default_timer()
                    print("{} :: Worker {: 2d}    ---    nach {:.2f} s abgeschlossen".format(datetime.now().strftime("%H:%M:%S"), idx, (end_time - start_time)))
                else:
                    print("{} :: Worker {: 2d}    ---   abgeschlossen".format(datetime.now().strftime("%H:%M:%S"), idx))
                # Fertig
                return

    except KeyboardInterrupt:
        if verbose:
            end_time = timeit.default_timer()
            print("{} :: Worker {: 2d}    ---    nach {:.2f} s abgeschlossen".format(datetime.now().strftime("%H:%M:%S"), idx, (end_time - start_time)))
        else:
            print("{} :: Worker {: 2d}    ---   abgeschlossen".format(datetime.now().strftime("%H:%M:%S"), idx))
        # Fertig
        return


def dispatch_testing(idx, args, global_model, summarywriter_path):
    """Die Worker Aufgabe für ein Testing"""

    try:
        summarywriter = SummaryWriter(summarywriter_path)

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
        local_episode = 0
        ep_rewards = []
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
                    hx = T.zeros((1, 512), dtype=T.float)
                    cx = T.zeros((1, 512), dtype=T.float)
                else: # Ansonsten wiederverwenden
                    hx = hx.detach()
                    cx = cx.detach()

            # Model
            action_logit_probability, action_judgement, hx, cx = local_model(local_state, hx, cx)

            # Policy
            policy = F.softmax(action_logit_probability, dim=1)

            # Action
            action = T.argmax(policy).item()

            # Action durchführen
            local_state, local_reward, local_done, info = env.step(action)
            ep_rewards.append(local_reward)
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

                latest_sum_reward = sum(ep_rewards)
                latest_avg_reward = latest_sum_reward / len(ep_rewards)

                # Tensorboard
                summarywriter.add_scalar("Tester-{}/X_Position".format(idx), info['x_pos'], local_episode)
                summarywriter.add_scalar("Tester-{}/Score".format(idx), info['score'], local_episode)
                summarywriter.add_scalar("Tester-{}/Coins".format(idx), info['coins'], local_episode)
                summarywriter.add_scalar("Tester-{}/Sum_Reward".format(idx), latest_sum_reward, local_episode)
                summarywriter.add_scalar("Tester-{}/Avg_Reward".format(idx), latest_avg_reward, local_episode)
                if info["flag_get"]: flag_get = 1
                else: flag_get = -1
                summarywriter.add_scalar("Tester-{}/Flag".format(idx), flag_get, local_episode)

                # Variablen zurückstetzten
                local_step = 0
                ep_rewards = []
                local_episode += 1
                actions.clear()

                # Env zurückstetzen
                local_state = env.reset()

            # zu numpy
            local_state = T.from_numpy(local_state)

    except KeyboardInterrupt:
        print("{} :: Runner {: 2d}    ---   abgeschlossen".format(datetime.now().strftime("%H:%M:%S"), idx))
        return
