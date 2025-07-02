# Documentation Complète - Infrastructure MLOps avec Kubernetes et MetalLB

## Table des Matières

1. [Vue d'ensemble du projet](#vue-densemble-du-projet)
2. [Architecture de l'infrastructure](#architecture-de-linfrastructure)
3. [Prérequis et configuration](#prérequis-et-configuration)
4. [Composants de l'infrastructure](#composants-de-linfrastructure)
5. [Installation et déploiement](#installation-et-déploiement)
6. [Configuration MetalLB](#configuration-metallb)
7. [Surveillance et monitoring](#surveillance-et-monitoring)
8. [Dépannage](#dépannage)
9. [Maintenance](#maintenance)

---

## Vue d'ensemble du projet

### Objectif

Cette infrastructure MLOps fournit une plateforme Kubernetes haute disponibilité avec équilibrage de charge, monitoring et services LoadBalancer pour les applications de machine learning en production.

### Technologies utilisées

- **Orchestration**: Ansible
- **Conteneurisation**: Kubernetes v1.31.4
- **Runtime de conteneurs**: containerd v1.7.27
- **Réseau**: Flannel CNI
- **Équilibrage de charge**: HAProxy + MetalLB
- **Monitoring**: Prometheus + Grafana
- **Ingress**: NGINX Ingress Controller

---

## Architecture de l'infrastructure

### Topologie réseau

```
Réseau: 10.110.190.0/24

10.110.190.21    - HAProxy Load Balancer
10.110.190.22    - Kubernetes Master 1
10.110.190.23    - Kubernetes Master 2  
10.110.190.24    - Kubernetes Master 3
10.110.190.25    - Kubernetes Worker 1
10.110.190.26    - Kubernetes Worker 2
10.110.190.27    - Kubernetes Worker 3
10.110.190.100-110 - Pool MetalLB (11 IPs disponibles)
```

### Flux de données

```
Client → HAProxy → Kubernetes Masters (API Server)
Client → HAProxy → Kubernetes Workers (Ingress NGINX)
MetalLB → Services LoadBalancer → Pods applicatifs
```

### Ports utilisés

- **API Kubernetes**: 6443
- **HAProxy Stats**: 8404
- **Ingress HTTP**: 80 → 32624
- **Ingress HTTPS**: 443 → 31316
- **Prometheus**: 30900
- **Grafana**: 30300

---

## Prérequis et configuration

### Système d'exploitation

- **OS**: Ubuntu 24.04.2 LTS
- **Kernel**: 6.8.0-53-generic
- **Architecture**: x86_64

### Configuration réseau requise

- Connectivité SSH entre le poste d'administration et tous les nœuds
- Résolution DNS ou fichier hosts configuré
- Ports réseau ouverts selon la matrice de ports

### Prérequis logiciels

```bash
# Sur le poste d'administration
sudo apt update
sudo apt install -y ansible python3-pip
pip3 install ansible

# Vérification version Ansible
ansible --version  # Version recommandée: 2.9+
```

---

## Composants de l'infrastructure

### 1. HAProxy Load Balancer

**Rôle**: Équilibrage de charge pour l'API Kubernetes et le trafic Ingress

**Configuration principale**:

- Mode TCP pour l'API Kubernetes
- Health checks automatiques
- Interface de statistiques sécurisée
- Distribution round-robin

**Fichier de configuration**: `ansible/roles/haproxy/templates/haproxy.cfg.j2`

### 2. Cluster Kubernetes

**Caractéristiques**:

- **Haute disponibilité**: 3 nœuds masters
- **Scalabilité**: 3 nœuds workers (extensible)
- **Version**: 1.31.4 (version LTS stable)
- **Control Plane Endpoint**: HAProxy (10.110.190.21:6443)

**Composants installés**:

- kubeadm, kubelet, kubectl
- containerd (runtime de conteneurs)
- Flannel (plugin réseau CNI)
- NGINX Ingress Controller

### 3. MetalLB Load Balancer

**Objectif**: Fournir des services LoadBalancer dans un environnement bare-metal

**Mode de fonctionnement**:

- **L2 Advertisement**: Annonce des IP via ARP
- **Pool d'adresses**: 10.110.190.100-110
- **Namespace**: metallb-system

**Version**: v0.14.8 (dernière version stable)

### 4. Stack de monitoring

**Composants**:

- **Prometheus**: Collecte et stockage des métriques
- **Grafana**: Visualisation et tableaux de bord
- **AlertManager**: Gestion des alertes
- **Node Exporter**: Métriques système
- **kube-state-metrics**: Métriques Kubernetes

---

## Installation et déploiement

### Structure du projet

```
/home/wassim/pfe/mlops/ansible/
├── ansible.cfg                    # Configuration Ansible
├── inventories/hosts.ini          # Inventaire des serveurs
├── group_vars/                    # Variables par groupe
│   ├── all.yml                   # Variables globales
│   ├── haproxy.yml               # Variables HAProxy
│   └── k8s-cluster.yml           # Variables Kubernetes
├── playbooks/                     # Playbooks de déploiement
│   ├── site.yml                  # Playbook principal
│   ├── install-metallb.yml       # Installation MetalLB
│   └── install-prometheus.yml    # Installation monitoring
└── roles/                         # Rôles Ansible
    ├── common/                   # Configuration système commune
    ├── haproxy/                  # Configuration HAProxy
    ├── kubernetes-master/        # Configuration masters K8s
    ├── kubernetes-worker/        # Configuration workers K8s
    └── metallb/                  # Configuration MetalLB
```

### Étapes d'installation

#### 1. Préparation de l'environnement

```bash
# Cloner ou accéder au projet
cd /home/wassim/pfe/mlops/ansible

# Vérifier la connectivité
ansible all -i inventories/hosts.ini -m ping

# Vérifier la configuration
ansible-inventory -i inventories/hosts.ini --list
```

#### 2. Déploiement complet de l'infrastructure

```bash
# Installation complète (HAProxy + Kubernetes + Prometheus + MetalLB)
ansible-playbook -i inventories/hosts.ini playbooks/site.yml

# Suivi des logs en temps réel
ansible-playbook -i inventories/hosts.ini playbooks/site.yml -v
```

#### 3. Déploiement par composant

```bash
# HAProxy uniquement
ansible-playbook -i inventories/hosts.ini playbooks/site.yml --tags haproxy

# Masters Kubernetes uniquement
ansible-playbook -i inventories/hosts.ini playbooks/site.yml --tags k8s-master

# Workers Kubernetes uniquement
ansible-playbook -i inventories/hosts.ini playbooks/site.yml --tags k8s-worker

# Prometheus uniquement
ansible-playbook -i inventories/hosts.ini playbooks/site.yml --tags prometheus

# MetalLB uniquement
ansible-playbook -i inventories/hosts.ini playbooks/site.yml --tags metallb
```

#### 4. Vérification post-installation

```bash
# Connexion au cluster
ssh user@10.110.190.22
kubectl get nodes
kubectl get pods --all-namespaces

# Vérification des services
kubectl get svc --all-namespaces
kubectl get ingress --all-namespaces
```

---

## Configuration MetalLB

### Problématique résolu

**Problème identifié par Zied**:

- Les services Prometheus/Grafana dans le namespace `monitoring` n'étaient pas de type LoadBalancer
- Les services ingress-nginx n'étaient pas accessibles depuis l'extérieur
- Impossibilité d'accéder aux interfaces de monitoring depuis l'extérieur du cluster

### Solution MetalLB

#### Installation

MetalLB est installé automatiquement avec le playbook principal, mais peut être installé séparément:

```bash
# Installation MetalLB uniquement
ansible-playbook -i inventories/hosts.ini playbooks/site.yml --tags metallb
```

#### Configuration du pool d'adresses IP

**IMPORTANT**: Attendre la plage d'adresses IP de Zied avant de configurer le pool.

Une fois la plage fournie, créer la configuration:

```yaml
# Fichier: metallb-config.yaml
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: default-pool
  namespace: metallb-system
spec:
  addresses:
  - [PLAGE_IP_FOURNIE_PAR_ZIED]
---
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: default-l2-adv
  namespace: metallb-system
spec:
  ipAddressPools:
  - default-pool
```

#### Application de la configuration

```bash
# Se connecter au premier master
ssh user@10.110.190.22

# Appliquer la configuration MetalLB
kubectl apply -f metallb-config.yaml

# Vérifier la configuration
kubectl get ipaddresspool -n metallb-system
kubectl get l2advertisement -n metallb-system
```

#### Création des services LoadBalancer

Une fois MetalLB configuré avec la plage IP, créer les services LoadBalancer:

```yaml
# Services LoadBalancer pour Prometheus/Grafana
apiVersion: v1
kind: Service
metadata:
  name: prometheus-server-lb
  namespace: monitoring
spec:
  type: LoadBalancer
  ports:
    - port: 9090
      targetPort: 9090
      name: http
  selector:
    app.kubernetes.io/name: prometheus
---
apiVersion: v1
kind: Service
metadata:
  name: grafana-lb
  namespace: monitoring
spec:
  type: LoadBalancer
  ports:
    - port: 3000
      targetPort: 3000
      name: http
  selector:
    app.kubernetes.io/name: grafana
---
# Service LoadBalancer pour Ingress NGINX
apiVersion: v1
kind: Service
metadata:
  name: ingress-nginx-controller-lb
  namespace: ingress-nginx
spec:
  type: LoadBalancer
  ports:
    - name: http
      port: 80
      targetPort: http
    - name: https
      port: 443
      targetPort: https
  selector:
    app.kubernetes.io/name: ingress-nginx
    app.kubernetes.io/component: controller
```

### Vérification du fonctionnement

```bash
# Vérifier les pods MetalLB
kubectl get pods -n metallb-system

# Vérifier les services LoadBalancer
kubectl get svc --all-namespaces --field-selector spec.type=LoadBalancer

# Vérifier l'attribution des IP externes
kubectl get svc -n monitoring
kubectl get svc -n ingress-nginx

# Tester l'accès externe
curl http://[IP_EXTERNE_GRAFANA]:3000
curl http://[IP_EXTERNE_PROMETHEUS]:9090
```

---

## Surveillance et monitoring

### Accès aux interfaces

#### Prometheus

- **URL interne**: <http://prometheus-server.monitoring.svc.cluster.local:9090>
- **NodePort**: http://[IP_WORKER]:30900
- **LoadBalancer**: http://[IP_EXTERNE]:9090 (après configuration MetalLB)

#### Grafana

- **URL interne**: <http://grafana.monitoring.svc.cluster.local:3000>
- **NodePort**: http://[IP_WORKER]:30300
- **LoadBalancer**: http://[IP_EXTERNE]:3000 (après configuration MetalLB)
- **Identifiants par défaut**: admin/admin

#### AlertManager

- **URL interne**: <http://alertmanager.monitoring.svc.cluster.local:9093>
- **LoadBalancer**: http://[IP_EXTERNE]:9093 (après configuration MetalLB)

### Métriques surveillées

#### Métriques système

- Utilisation CPU, mémoire, disque
- Charge système et uptime
- Métriques réseau (trafic, erreurs)

#### Métriques Kubernetes

- État des pods, services, déploiements
- Utilisation des ressources par namespace
- Événements du cluster

#### Métriques applicatives

- Latence des requêtes
- Taux d'erreur
- Throughput

### Configuration des alertes

Les alertes par défaut incluent:

- Nœuds Kubernetes indisponibles
- Pods en échec
- Utilisation haute des ressources
- Problèmes de connectivité réseau

---

## Dépannage

### Problèmes courants

#### 1. Échec de connexion SSH

```bash
# Vérifier la connectivité
ping 10.110.190.22

# Vérifier la configuration SSH
ssh -v user@10.110.190.22

# Vérifier les clés SSH
ssh-copy-id user@10.110.190.22
```

#### 2. Problèmes de déploiement Kubernetes

```bash
# Vérifier l'état du cluster
kubectl get nodes
kubectl get pods --all-namespaces

# Logs des composants système
kubectl logs -n kube-system [POD_NAME]

# Vérifier la configuration kubeadm
sudo kubeadm config view
```

#### 3. Problèmes MetalLB

```bash
# Vérifier les pods MetalLB
kubectl get pods -n metallb-system

# Logs MetalLB
kubectl logs -n metallb-system -l app=metallb

# Vérifier la configuration
kubectl describe ipaddresspool -n metallb-system
```

#### 4. Problèmes HAProxy

```bash
# Vérifier le statut HAProxy
ssh user@10.110.190.21
sudo systemctl status haproxy

# Logs HAProxy
sudo journalctl -u haproxy -f

# Interface de statistiques
curl http://10.110.190.21:8404/stats
```

### Commandes utiles de diagnostic

```bash
# État général du cluster
kubectl cluster-info
kubectl get nodes -o wide
kubectl get pods --all-namespaces

# Événements du cluster
kubectl get events --sort-by='.lastTimestamp'

# Utilisation des ressources
kubectl top nodes
kubectl top pods --all-namespaces

# Configuration réseau
kubectl get svc --all-namespaces
kubectl get ingress --all-namespaces
kubectl get endpoints --all-namespaces
```

---

## Maintenance

### Sauvegardes

#### 1. Sauvegarde etcd

```bash
# Se connecter au master
ssh user@10.110.190.22

# Créer une sauvegarde etcd
sudo ETCDCTL_API=3 etcdctl snapshot save backup.db \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/healthcheck-client.crt \
  --key=/etc/kubernetes/pki/etcd/healthcheck-client.key
```

#### 2. Sauvegarde des configurations

```bash
# Sauvegarder les ressources Kubernetes
kubectl get all --all-namespaces -o yaml > cluster-backup.yaml

# Sauvegarder les ConfigMaps et Secrets
kubectl get configmaps --all-namespaces -o yaml > configmaps-backup.yaml
kubectl get secrets --all-namespaces -o yaml > secrets-backup.yaml
```

### Mises à jour

#### 1. Mise à jour Kubernetes

```bash
# Planifier la mise à jour
kubeadm version
kubeadm upgrade plan

# Appliquer la mise à jour (sur chaque nœud)
sudo kubeadm upgrade apply v1.31.5
sudo apt update && sudo apt upgrade kubelet kubectl
sudo systemctl restart kubelet
```

#### 2. Mise à jour MetalLB

```bash
# Vérifier la version actuelle
kubectl get pods -n metallb-system -o yaml | grep image:

# Mettre à jour vers une nouvelle version
kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.15.0/config/manifests/metallb-native.yaml
```

### Surveillance continue

#### 1. Monitoring automatisé

- Configurer des alertes Prometheus pour les métriques critiques
- Mettre en place des checks de santé automatiques
- Surveiller l'utilisation des ressources

#### 2. Logs centralisés

- Déployer une stack ELK (Elasticsearch, Logstash, Kibana)
- Configurer Fluentd pour la collecte des logs
- Centraliser les logs système et applicatifs

---

## Conclusion

Cette infrastructure MLOps fournit une base solide pour déployer et gérer des applications de machine learning en production. Avec MetalLB, le problème d'accès externe aux services de monitoring est résolu, permettant une surveillance complète de l'infrastructure depuis l'extérieur du cluster.

### Points clés de réussite

- **Haute disponibilité**: 3 masters, 3 workers
- **Équilibrage de charge**: HAProxy + MetalLB
- **Monitoring complet**: Prometheus + Grafana
- **Sécurité**: RBAC, containerd sécurisé
- **Scalabilité**: Architecture extensible

### Prochaines étapes recommandées

1. Configuration du pool IP MetalLB avec la plage fournie par Zied
2. Mise en place des services LoadBalancer pour l'accès externe
3. Configuration des alertes personnalisées
4. Déploiement d'applications ML de test
5. Mise en place de pipelines CI/CD

---

*Documentation générée le 2 juillet 2025*  
*Version: 1.0*  
*Auteur: Wassim Mezrani*
