# Zen Note Frontend - Next.js UI

A modern, responsive React frontend for the Zen Note AI-powered knowledge search system.

## 🎨 Design Features

- **Clean, minimal interface** with modern Inter typography
- **Responsive design** with TailwindCSS
- **Professional typography** using Inter font family
- **Accessibility-first** with proper ARIA labels and keyboard navigation
- **Real-time interaction** with loading states and error handling

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18+ 
- **npm** or **yarn**
- **Backend API** running on `localhost:8000` (see `../backend/README.md`)

### Installation & Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

The application will be available at **http://localhost:3000**

## 🏗️ Architecture

### Component Structure
```
src/
├── app/
│   ├── layout.tsx         # Root layout with fonts and metadata
│   ├── page.tsx           # Main chat interface
│   └── globals.css        # Global styles and Zen Note theme
```

### Key Features

#### 🎯 **Main Interface (`page.tsx`)**
- **Question Input**: Large, accessible text input with placeholder
- **Submit Button**: Disabled state when loading or empty input
- **Answer Display**: Formatted response area with error handling
- **Keyboard Support**: Enter key to submit questions

#### 🎨 **Design System**
- **Colors**: Green theme (`#075907`, `#f0f4f0`, `#618961`)
- **Typography**: Inter font family for all text
- **Layout**: Centered content with responsive padding
- **Components**: Consistent spacing and border radius

#### ♿ **Accessibility**
- **Screen Reader Support**: Proper ARIA labels and semantic HTML
- **Keyboard Navigation**: Full keyboard accessibility with `tabIndex`
- **Focus Management**: Clear focus indicators
- **Color Contrast**: WCAG AA compliant color choices

## 🔌 API Integration

### Backend Communication
```typescript
// POST request to RAG endpoint
const response = await fetch('http://localhost:8000/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question }),
});

const data = await response.json();
```

### Error Handling
- **Network Errors**: Connection failure messages
- **API Errors**: Backend error responses
- **Loading States**: Visual feedback during processing
- **Empty Inputs**: Button disabled for empty questions

### CORS Configuration
The backend is configured to accept requests from:
- `http://localhost:3000` (development)
- `http://127.0.0.1:3000` (alternative localhost)

## 📱 Responsive Design

### Breakpoints
- **Desktop**: Full padding and wide layout
- **Mobile**: Reduced padding, stacked components
- **Touch**: Larger tap targets for buttons

### Mobile-First Approach
```css
/* Mobile base styles */
.px-10 { padding-left: 2.5rem; padding-right: 2.5rem; }

/* Desktop enhancement */
.px-40 { padding-left: 10rem; padding-right: 10rem; }
```

## 🎨 Styling Guide

### Color Palette
```css
/* Primary Colors */
--zen-primary: #075907;      /* Dark green for CTAs */
--zen-light: #f0f4f0;        /* Light green for backgrounds */
--zen-border: #dbe6db;       /* Green border color */
--zen-text: #618961;         /* Green text for placeholders */
--zen-dark: #111811;         /* Dark text for content */
```

### Typography Scale
```css
/* Heading */
text-[28px] font-bold        /* Main title */

/* Body */
text-base font-normal        /* Regular text */
text-sm font-medium         /* Navigation links */
text-lg font-semibold       /* Answer heading */
```

### Component Patterns
```css
/* Input Fields */
rounded-xl border border-[#dbe6db] p-[15px]

/* Buttons */
rounded-full bg-[#075907] text-white px-4 h-10

/* Cards */
bg-[#f8f9fa] border border-[#dbe6db] rounded-xl p-4
```

## 🔧 Configuration

### Environment Variables
Create `.env.local` for custom configuration:
```bash
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000

# App Configuration
NEXT_PUBLIC_APP_NAME="Zen Note"
```

### Next.js Configuration
Current setup in `next.config.ts`:
```typescript
const nextConfig = {
  experimental: {
    optimizePackageImports: ['@icons/react']
  }
};
```

## 🧪 Testing & Development

### Development Features
- **Hot Reload**: Instant updates during development
- **Error Overlay**: Clear error messages in development
- **TypeScript**: Full type checking for better DX

### Testing Checklist
- [ ] Question submission works with backend
- [ ] Loading states display correctly  
- [ ] Error handling for network failures
- [ ] Keyboard navigation (Tab, Enter)
- [ ] Responsive design on mobile
- [ ] Accessibility with screen reader

### Debug Mode
```bash
# Enable verbose logging
npm run dev -- --debug

# Check network requests in browser DevTools
# Monitor console for errors or warnings
```

## 🚀 Deployment

### Build Process
```bash
# Create production build
npm run build

# Verify build works locally
npm start
```

### Production Considerations
- **Environment Variables**: Set production API URL
- **Performance**: Automatic code splitting and optimization
- **SEO**: Meta tags configured in layout.tsx
- **Fonts**: Google Fonts optimized loading

### Deploy Options
```bash
# Vercel (recommended)
vercel deploy

# Netlify
netlify deploy --build

# Docker
docker build -t zen-note-frontend .
```

## 📊 Performance

### Optimization Features
- **Next.js 15**: Latest performance improvements
- **Font Optimization**: Automatic Google Fonts optimization
- **Code Splitting**: Automatic route-based splitting
- **Image Optimization**: Built-in Next.js image optimization

### Bundle Analysis
```bash
# Analyze bundle size
npm install -g @next/bundle-analyzer
npm run build
npm run analyze
```

## 🔮 Future Enhancements

### Planned Features
- **Chat History**: Save and restore previous conversations
- **Source Attribution**: Display which documents were used
- **Advanced Filtering**: Filter by document type or date
- **Themes**: Dark mode and custom color schemes
- **Export**: Save answers as markdown or PDF

### Technical Improvements
- **PWA Support**: Offline capability
- **Real-time Updates**: WebSocket connection for live responses
- **Advanced Error Boundary**: Better error recovery
- **Performance Monitoring**: Real user metrics

## 🔗 Integration

### Backend Integration
This frontend is designed to work with the FastAPI backend:
- **Endpoint**: `POST /ask` for question answering
- **Health Check**: `GET /health` for system status  
- **Search**: `GET /search` for debugging retrieval

### Future Integrations
- **Authentication**: User login and session management
- **Document Upload**: Direct file upload interface
- **Settings**: Configuration panel for RAG parameters

---

**Ready to ask questions?** Start the dev server and visit http://localhost:3000!
