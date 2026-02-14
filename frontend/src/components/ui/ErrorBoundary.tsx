"use client";

import { Component, type ReactNode } from "react";
import { Card } from "./Card";
import { Button } from "./Button";
import { AlertTriangle, RotateCcw } from "lucide-react";

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-[400px] flex items-center justify-center p-8">
          <Card className="max-w-md w-full text-center">
            <div className="flex justify-center mb-4">
              <div className="h-12 w-12 rounded-2xl bg-danger-light flex items-center justify-center">
                <AlertTriangle className="h-6 w-6 text-danger" />
              </div>
            </div>
            <h3 className="text-lg font-semibold text-text mb-2">
              Something went wrong
            </h3>
            <p className="text-sm text-text-secondary mb-6">
              {this.state.error?.message || "An unexpected error occurred."}
            </p>
            <Button
              variant="secondary"
              onClick={() => this.setState({ hasError: false, error: null })}
              icon={<RotateCcw className="h-4 w-4" />}
            >
              Try Again
            </Button>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}
