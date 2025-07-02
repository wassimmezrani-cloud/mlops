# Infrastructure MLOps avec Kubernetes et MetalLB

Infrastructure MLOps automatisée avec Ansible pour déployer un cluster Kubernetes haute disponibilité avec équilibrage de charge et monitoring.

## Architecture

- **HAProxy**: Load balancer externe (10.110.190.21)
- **3 Masters**: Control plane Kubernetes (10.110.190.22-24)  
- **3 Workers**: Nœuds de traitement (10.110.190.25-27)
- **MetalLB**: Load balancer interne pour services
- **Prometheus/Grafana**: Stack de monitoring

## Installation

### Prérequis

- Ansible 2.9+
- Accès SSH aux serveurs
- Ubuntu 24.04 LTS

### Déploiement

```bash
cd ansible

# Installation complète
ansible-playbook -i inventories/hosts.ini playbooks/site.yml

# MetalLB uniquement
ansible-playbook -i inventories/hosts.ini playbooks/site.yml --tags metallb
```

## Configuration MetalLB

En attente de la plage IP de Zied pour configurer le pool d'adresses.

Voir `GUIDE_DEPLOIEMENT.md` pour les étapes détaillées.

## Documentation

- Documentation technique: `DOCUMENTATION_TECHNIQUE_COMPLETE.md`
- PDF: `documentation-technique.pdf`
- Guide déploiement: `GUIDE_DEPLOIEMENT.md`

## Support

Projet développé par Wassim Mezrani pour l'infrastructure MLOps.
