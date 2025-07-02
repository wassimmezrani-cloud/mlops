# Guide de Déploiement Production - Infrastructure MLOps

## Prérequis de Production

### Sécurité
```bash
# 1. Générer les mots de passe sécurisés
ansible-vault create group_vars/vault.yml

# Contenu du vault:
vault_grafana_admin_password: "secure_password_here"
vault_haproxy_stats_password: "secure_password_here"
```

### Vérifications SSH
```bash
# 2. Configurer les clés SSH
ssh-copy-id user@10.110.190.21
ssh-copy-id user@10.110.190.22
# ... pour tous les serveurs

# 3. Tester la connectivité
ansible all -i inventories/hosts.ini -m ping
```

## Déploiement Infrastructure

### Étape 1: Déploiement Complet
```bash
cd /home/wassim/pfe/mlops/ansible

# Déploiement avec vault
ansible-playbook -i inventories/hosts.ini playbooks/site.yml --ask-vault-pass

# Ou avec fichier de mot de passe vault
echo "vault_password" > .vault_pass
chmod 600 .vault_pass
ansible-playbook -i inventories/hosts.ini playbooks/site.yml --vault-password-file .vault_pass
```

### Étape 2: Configuration MetalLB
```bash
# Attendre la plage IP de Zied, puis:
ansible-playbook -i inventories/hosts.ini playbooks/metallb-configure.yml \
  -e "metallb_ip_pool_range=IP_RANGE_DE_ZIED" \
  --vault-password-file .vault_pass
```

### Étape 3: Activation LoadBalancer Services
```bash
ansible-playbook -i inventories/hosts.ini playbooks/services-loadbalancer.yml \
  --vault-password-file .vault_pass
```

## Vérifications Post-Déploiement

### Cluster Kubernetes
```bash
ssh user@10.110.190.22
kubectl get nodes -o wide
kubectl get pods --all-namespaces
kubectl get svc --all-namespaces
```

### Services MetalLB
```bash
kubectl get svc --all-namespaces --field-selector spec.type=LoadBalancer
kubectl get ipaddresspool,l2advertisement -n metallb-system
```

### Monitoring
```bash
# Accès Prometheus: http://EXTERNAL_IP:9090
# Accès Grafana: http://EXTERNAL_IP:3000 (admin/VAULT_PASSWORD)
# HAProxy Stats: http://10.110.190.21:8404/stats (admin/VAULT_PASSWORD)
```

## Sécurité Production

### Variables Sensibles
Toutes les variables sensibles doivent être dans vault.yml:
- vault_grafana_admin_password
- vault_haproxy_stats_password
- Futurs certificats TLS

### SSH Sécurisé
- host_key_checking = True (activé)
- Clés SSH uniquement (pas de mots de passe)
- Vérification des fingerprints

## Maintenance

### Sauvegardes
```bash
# Sauvegarde etcd
kubectl get all --all-namespaces -o yaml > cluster-backup.yaml

# Sauvegarde configuration Ansible
tar -czf ansible-config-backup.tar.gz ansible/
```

### Surveillance
- Prometheus: Métriques système et cluster
- Grafana: Tableaux de bord de monitoring
- HAProxy: Stats et health checks

## Résolution de Problèmes

### Logs Ansible
```bash
tail -f /tmp/ansible.log
```

### Logs Kubernetes
```bash
kubectl logs -n kube-system kube-apiserver-MASTER_NAME
kubectl logs -n metallb-system -l app=metallb
```

### Tests de Connectivité
```bash
# Test HAProxy
curl -I http://10.110.190.21:8404/stats

# Test API Kubernetes via HAProxy
kubectl --server=https://10.110.190.21:6443 get nodes
```