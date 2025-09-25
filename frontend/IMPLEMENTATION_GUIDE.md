# Diagnogenie Design System - Implementation Guide

This guide shows how to implement the Diagnogenie design system in your React components.

## Quick Start

1. Import the design system CSS in your main CSS file:
```css
@import './design-system.css';
```

2. Import components from the design system:
```jsx
import { Button, Heading1, Alert, ChatBubble } from './components/DesignSystem';
```

## Component Integration Examples

### Updating Your Chatbot Component

Here's how to integrate the design system into your existing `Chatbot.jsx`:

```jsx
// Before (example)
<div className="bg-blue-500 text-white p-4 rounded">
  User message
</div>

// After (with design system)
<ChatBubble type="user">
  User message
</ChatBubble>
```

### Updating Image Upload Component

For your `ImageUpload.jsx`:

```jsx
import { ImageUpload, Alert } from './DesignSystem';

const ImageUploadComponent = () => {
  const [uploadStatus, setUploadStatus] = useState('default');
  
  const handleFileSelect = (file) => {
    // Your upload logic
    setUploadStatus('success');
  };

  return (
    <div>
      <ImageUpload 
        onFileSelect={handleFileSelect}
        status={uploadStatus}
      />
      {uploadStatus === 'success' && (
        <Alert type="success">
          Image uploaded successfully!
        </Alert>
      )}
    </div>
  );
};
```

### Language Support Integration

Add multilingual support to any component:

```jsx
import { LanguageToggle } from './DesignSystem';

const App = () => {
  const [language, setLanguage] = useState('en');
  
  const handleLanguageChange = (newLanguage) => {
    setLanguage(newLanguage);
    // Update your i18n context or state
  };

  return (
    <div>
      <LanguageToggle 
        currentLanguage={language}
        onLanguageChange={handleLanguageChange}
      />
      {/* Rest of your app */}
    </div>
  );
};
```

## Color Usage Guidelines

### CSS Classes (Tailwind)
```jsx
// Primary colors
<div className="bg-primary text-white">Primary action</div>
<div className="bg-primary-dark text-white">Navigation header</div>
<div className="bg-primary-light">Subtle background</div>

// Status colors
<div className="text-secondary-success">Success message</div>
<div className="text-secondary-warning">Warning message</div>
<div className="text-secondary-error">Error message</div>

// Neutral colors
<div className="text-neutral-dark-gray">Primary text</div>
<div className="text-neutral-medium-gray">Secondary text</div>
<div className="border-neutral-light-gray">Borders</div>
```

### CSS Variables
```css
.custom-component {
  background-color: var(--primary-blue);
  color: var(--background-white);
  border-radius: var(--radius-button);
  padding: var(--space-m);
}
```

## Typography Implementation

### Using Design System Typography
```jsx
import { Heading1, Heading2, BodyText, Caption } from './DesignSystem';

const ExampleComponent = () => (
  <div>
    <Heading1>Medical Dashboard</Heading1>
    <Heading2>Patient Information</Heading2>
    <BodyText>Patient details and medical history...</BodyText>
    <Caption>Last updated: 2 hours ago</Caption>
  </div>
);
```

### Using Tailwind Classes
```jsx
<h1 className="text-h1 font-bold text-neutral-dark-gray">Page Title</h1>
<p className="text-body text-neutral-dark-gray">Body content</p>
<span className="text-caption font-medium text-neutral-medium-gray">Metadata</span>
```

## Spacing System

### Margin and Padding
```jsx
// Using Tailwind classes
<div className="p-m mb-l">Content with medium padding and large bottom margin</div>
<div className="mt-xl">Large top margin</div>

// Using CSS variables
<div style={{ padding: 'var(--space-m)', marginBottom: 'var(--space-l)' }}>
  Content
</div>
```

### Spacer Component
```jsx
import { Spacer } from './DesignSystem';

<div>
  <h1>Title</h1>
  <Spacer size="l" />
  <p>Content with large spacing above</p>
</div>
```

## Responsive Design

### Grid System
```jsx
import { Container } from './DesignSystem';

<Container>
  <div className="grid grid-cols-1 tablet:grid-cols-2 desktop:grid-cols-3 gap-m">
    <div>Column 1</div>
    <div>Column 2</div>
    <div>Column 3</div>
  </div>
</Container>
```

### Responsive Utilities
```jsx
<div className="p-s tablet:p-m desktop:p-l">
  Responsive padding
</div>
```

## Accessibility Features

### Focus States
All interactive components automatically include proper focus states:
```jsx
<Button>Accessible button with focus ring</Button>
```

### Color Contrast
The design system ensures WCAG AA compliance:
```jsx
<div className="bg-primary text-white">High contrast text</div>
```

### Screen Reader Support
```jsx
<Button aria-label="Upload medical image">
  📁
</Button>
```

## Animation Usage

### CSS Classes
```jsx
<div className="animate-fade-in">Fading in content</div>
<div className="animate-slide-in-bottom">Sliding up modal</div>
```

### Reduced Motion Support
The design system automatically respects user preferences:
```css
@media (prefers-reduced-motion: reduce) {
  /* Animations are automatically disabled */
}
```

## Migration Checklist

When updating existing components to use the design system:

### ✅ Colors
- [ ] Replace custom colors with design system variables
- [ ] Use semantic color names (primary, success, error)
- [ ] Ensure proper contrast ratios

### ✅ Typography  
- [ ] Use design system font sizes and weights
- [ ] Implement proper line heights
- [ ] Add Hindi font support where needed

### ✅ Spacing
- [ ] Use the 8px spacing scale
- [ ] Replace arbitrary spacing with system values
- [ ] Ensure consistent component spacing

### ✅ Components
- [ ] Replace custom buttons with design system buttons
- [ ] Update form elements to use system styles
- [ ] Implement consistent card and container layouts

### ✅ Accessibility
- [ ] Add proper focus states
- [ ] Include ARIA labels where needed
- [ ] Test with screen readers
- [ ] Ensure keyboard navigation

## Best Practices

### Do's ✅
- Use semantic HTML elements
- Follow the spacing scale consistently
- Maintain color contrast ratios
- Test with different screen sizes
- Include proper loading states

### Don'ts ❌
- Don't override design system colors arbitrarily
- Avoid mixing font weights within components  
- Don't skip accessibility attributes
- Avoid using hardcoded pixel values for spacing

## Development Workflow

1. **Design First**: Check the design system before creating custom styles
2. **Component Reuse**: Use existing components when possible
3. **Extend Thoughtfully**: Create new variants through the design system
4. **Test Accessibility**: Always verify focus states and screen reader compatibility
5. **Document Changes**: Update this guide when adding new components

## Support

For questions or suggestions about the design system:
- Check the main documentation in `DESIGN_SYSTEM.md`
- Review component examples in `DesignSystem.jsx`
- Test implementations with the showcase component

Remember: The design system is a living document. Update it as your application evolves while maintaining consistency and usability.