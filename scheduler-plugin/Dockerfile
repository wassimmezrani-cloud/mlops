# Multi-stage build for ML-Scheduler Plugin
FROM golang:1.21-alpine AS builder

# Install dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Set working directory
WORKDIR /workspace

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY cmd/ cmd/
COPY pkg/ pkg/

# Build the scheduler plugin
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -a -installsuffix cgo \
    -ldflags="-w -s -X main.version=1.0.0 -X main.buildDate=$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    -o ml-scheduler \
    ./cmd/main.go

# Production stage
FROM gcr.io/distroless/static:nonroot

# Copy ca-certificates from builder
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy timezone data
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# Copy the scheduler binary
COPY --from=builder /workspace/ml-scheduler /usr/local/bin/ml-scheduler

# Use nonroot user
USER 65534:65534

# Expose ports
EXPOSE 8080 8081 10259

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["/usr/local/bin/ml-scheduler", "--help"]

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/ml-scheduler"]