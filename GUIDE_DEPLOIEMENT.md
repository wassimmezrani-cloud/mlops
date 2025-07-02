# Guide de Déploiement Infrastructure MLOps

## Étape 1: Installation Infrastructure Complète

```bash
cd /home/wassim/pfe/mlops/ansible
ansible-playbook -i inventories/hosts.ini playbooks/site.yml
```

## Étape 2: Configuration MetalLB (Après réception plage IP)

```bash
ansible-playbook -i inventories/hosts.ini playbooks/metallb-configure.yml \
  -e "metallb_ip_pool_range=PLAGE_IP_ZIED"
```

## Étape 3: Activation Services LoadBalancer

```bash
ansible-playbook -i inventories/hosts.ini playbooks/services-loadbalancer.yml
```

## Vérification

```bash
# Connexion au cluster
ssh user@10.110.190.22

# Vérifier services LoadBalancer
kubectl get svc --all-namespaces --field-selector spec.type=LoadBalancer

# Vérifier accès externe
curl http://EXTERNAL_IP:9090  # Prometheus
curl http://EXTERNAL_IP:3000  # Grafana
```

## Résultat Attendu

- Prometheus accessible via IP externe port 9090
- Grafana accessible via IP externe port 3000
- AlertManager accessible via IP externe port 9093
- Ingress NGINX accessible via IP externe ports 80/443

## Status Actuel

✅ MetalLB installation prête
⏳ Attente plage IP de Zied
⏳ Configuration pool IP en attente
⏳ Conversion services en attente
