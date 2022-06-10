import torch
from utilities.adv_utils import apply_sobel
from kd_lib.losses import dml_Loss, collate_loss
from kd_lib import losses as kd_losses
from train.adversarial_loss import trades_loss, madry_loss

def adv_training(
    args,
    iteration,
    num_batches,
    model1,
    model2,
    data1,
    data2,
    target,
    optimizer_m1,
    optimizer_m2,
    m1_train_loss,
    m1_correct,
    m2_train_loss,
    m2_correct,
    total,
    writer
):

    if args.adv_option == 'v1' :
        m1_out = model1(data1)
        m2_out = model2(data2)
        l_ce_m2 = kd_losses.cross_entropy(m2_out, target)
        dml_loss_m1 = dml_Loss(args, model1.feat, [ft.detach() for ft in model2.feat],
                           m1_out, m2_out.detach())
        dml_loss_m2 = dml_Loss(args, model2.feat, [ft.detach() for ft in model1.feat],
                               m2_out, m1_out.detach())
    elif args.adv_option == 'v2' :
        m2_out = model2(data2)
        l_ce_m2 = kd_losses.cross_entropy(m2_out, target)

    elif args.adv_option == 'v3' :
        m2_out = model2(data2)
        l_ce_m2 = kd_losses.cross_entropy(m2_out, target)

    l_ce_m1 = 0
    if 'madry' in args.adv_loss_type :
        loss_adv, m1_adv_out, x_adv = madry_loss(
            model1,
            data1,
            target,
            optimizer_m1,
            step_size=args.step_size,
            epsilon=args.epsilon,
            perturb_steps=args.num_steps,
        )

    elif 'trades' in args.adv_loss_type :
        loss_adv, m1_adv_out, x_adv = trades_loss(
            model1,
            data1,
            target,
            optimizer_m1,
            step_size=args.step_size,
            epsilon=args.epsilon,
            perturb_steps=args.num_steps,
            beta=args.beta,
            distance=args.distance,
        )
    else:
        print("No adversarial training selected")
        exit()

    if args.adv_option == 'v2' :
        dml_loss_m1 = dml_Loss(args, model1.feat, [ft.detach() for ft in model2.feat],
                           m1_adv_out, m2_out.detach())
    elif args.adv_option == 'v3' :
        dml_loss_m1 = dml_Loss(args, model1.feat, [ft.detach() for ft in model2.feat],
                           m1_adv_out, m2_out.detach())
        dml_loss_m2 = dml_Loss(args, model2.feat, [ft.detach() for ft in model1.feat],
                               m2_out, m1_adv_out.detach())
    elif args.adv_option == 'v4' :
        x_adv_sob = apply_sobel(args, x_adv)
        m2_adv_out = model2(x_adv_sob)
        l_ce_m2 = kd_losses.cross_entropy(m2_adv_out, target)
        dml_loss_m1 = dml_Loss(args, model1.feat, [ft.detach() for ft in model2.feat],
                           m1_adv_out, m2_adv_out.detach())
    elif args.adv_option == 'v5' :
        x_adv_sob = apply_sobel(args, x_adv)
        m2_adv_out = model2(x_adv_sob)
        l_ce_m2 = kd_losses.cross_entropy(m2_adv_out, target)
        dml_loss_m1 = dml_Loss(args, model1.feat, [ft.detach() for ft in model2.feat],
                           m1_adv_out, m2_adv_out.detach())
        dml_loss_m2 = dml_Loss(args, model2.feat, [ft.detach() for ft in model1.feat],
                               m2_adv_out, m1_adv_out.detach())

    loss_m1_dict = collate_loss(args, loss_ce=l_ce_m1, loss_dml=dml_loss_m1.loss_dml, loss_adv = loss_adv, m1=True)
    loss_m2_dict = collate_loss(args, loss_ce=l_ce_m2, loss_dml=None if args.adv_option=='v2' or args.adv_option=='v4' else dml_loss_m2.loss_dml, loss_adv = 0, m1=False)

    loss_m1 = loss_m1_dict['loss']
    loss_m2 = loss_m2_dict['loss']

    loss_m1.backward()
    optimizer_m1.step()

    loss_m2.backward()
    optimizer_m2.step()

    for loss_name, loss_item in loss_m1_dict.items():
        writer.add_scalar('Model1 losses/{}'.format(loss_name), loss_item,
                          global_step=iteration)
    for loss_name, loss_item in loss_m2_dict.items():
        writer.add_scalar('Model2 losses/{}'.format(loss_name), loss_item,
                          global_step=iteration)

    m1_train_loss += loss_m1.data.item()
    m2_train_loss += loss_m2.data.item()

    _, predicted_m1 = torch.max(m1_adv_out, 1)
    _, predicted_m2 = torch.max(m2_adv_out.data, 1) if args.adv_option=='v4' or args.adv_option=='v5' else torch.max(m2_out.data, 1)

    m1_correct += predicted_m1.eq(target.data).cpu().float().sum()
    m2_correct += predicted_m2.eq(target.data).cpu().float().sum()
    total += target.size(0)

    m1_train_loss /= num_batches + 1
    m1_acc = 100.0 * m1_correct / total

    m2_train_loss /= num_batches + 1
    m2_acc = 100.0 * m2_correct / total

    return m1_train_loss, m1_acc, m1_correct, m2_train_loss, m2_acc, m2_correct