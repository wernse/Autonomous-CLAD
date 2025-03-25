from copy import deepcopy

import torch


def combine_teachers(cpt, buf_teacher_logits, task_id_nominal, buf_labels, buf_logits, batch_size_buf, combo, epoch):
    counter = 0
    counter2 = 0
    counter3 = 0
    counter4 = 0
    try:
        buf_logits_tmp = deepcopy(buf_logits)
    except:
        buf_logits_tmp = buf_logits
    for t in range(task_id_nominal):
        start_KD = t * cpt
        end_KD = (t + 1) * cpt

        relevant_student_org = torch.tensor(
            [idx // cpt == t for idx in range(buf_teacher_logits.size(1))],
            device=buf_labels.device)

        task_labels = buf_labels[t * batch_size_buf:(t + 1) * batch_size_buf]
        prev_score = buf_logits_tmp[t * batch_size_buf:(t + 1) * batch_size_buf]
        score = buf_teacher_logits[t * batch_size_buf:(t + 1) * batch_size_buf]

        org_teacher_score_logits = prev_score[:, start_KD:end_KD]
        special_teacher_score_logits = score[:, start_KD:end_KD]

        if combo == 0:
            score_teacher = org_teacher_score_logits if epoch % 2 == 0 else special_teacher_score_logits
            transplant_exc_teacher = prev_score[:, ~relevant_student_org]
            max_values = transplant_exc_teacher.max(1).values
            max_teacher_values = score_teacher.max(1).values
            mask = (max_values > max_teacher_values).unsqueeze(1).repeat(1, prev_score.shape[1])
            coeff = 0.75 * max_teacher_values / max_values
            coeff = coeff.unsqueeze(1).repeat(1, prev_score.shape[1])
            if mask.sum() > 0:
                true_indices = mask.nonzero(as_tuple=True)[0]
                prev_score[mask] *= coeff[mask]
                counter += 1
            buf_logits_tmp[t * batch_size_buf:(t + 1) * batch_size_buf, start_KD:end_KD] = score_teacher

        if combo == 1:
            if epoch % 2 == 0:
                score_teacher = special_teacher_score_logits
                transplant_exc_teacher = prev_score[:, ~relevant_student_org]
                max_values = transplant_exc_teacher.max(1).values
                max_teacher_values = score_teacher.max(1).values
                mask = (max_values > max_teacher_values).unsqueeze(1).repeat(1, prev_score.shape[1])
                coeff = 0.75 * max_teacher_values / max_values
                coeff = coeff.unsqueeze(1).repeat(1, prev_score.shape[1])
                if mask.sum() > 0:
                    true_indices = mask.nonzero(as_tuple=True)[0]
                    prev_score[mask] *= coeff[mask]
                    counter += 1
                buf_logits_tmp[t * batch_size_buf:(t + 1) * batch_size_buf, start_KD:end_KD] = score_teacher

        if combo == 2:
            if epoch % 2 == 0:
                buf_logits_tmp[t * batch_size_buf:(t + 1) * batch_size_buf, start_KD:end_KD] = special_teacher_score_logits


        if combo == 3:
            _, pred_org_teacher = torch.max(prev_score.data, 1)
            _, pred_special_teacher = torch.max(score.data, 1)

            org_correct = pred_org_teacher == task_labels
            special_correct = pred_special_teacher == task_labels

            for i in range(len(org_correct)):
                if org_correct[i] and special_correct[i]:
                    counter += 1
                    # Both correct, leave as is
                    pass
                elif org_correct[i] and not special_correct[i]:
                    counter2 += 1
                    # Original correct, special wrong, leave as is
                    pass
                elif not org_correct[i] and special_correct[i]:
                    counter3 += 1
                    # Original wrong, special correct, replace with special
                    buf_logits_tmp[t * batch_size_buf + i, start_KD:end_KD] = special_teacher_score_logits[i]
                else:
                    counter4 += 1
                    pass
                    # Both wrong, average the logits
                    # avg_teacher_logits = (org_teacher_score_logits[i] + special_teacher_score_logits[i]) / 2
                    # buf_logits_tmp[t * batch_size_buf + i, start_KD:end_KD] = avg_teacher_logits

        if combo == 4:
            avg_teacher_logits = (org_teacher_score_logits + special_teacher_score_logits) / 2
            buf_logits_tmp[t * batch_size_buf:(t + 1) * batch_size_buf, start_KD:end_KD] = avg_teacher_logits

    return buf_logits_tmp, counter, counter2, counter3, counter4
